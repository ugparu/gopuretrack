package puretrack

import (
	"math"
	"slices"

	"github.com/ugparu/gopuretrack/utils"
)

// detInfo is the per-detection working record threaded through the pipeline.
// Pre-computing it once avoids recomputing per-detection derived values
// (center-plus-size box, visibility overlap) in every association pass.
type detInfo[T Detection] struct {
	det     T
	xywh    [4]float64
	conf    float64
	emb     []float64
	class   int
	detID   int
	overlap float64 // max fractional overlap by any other detection this frame
}

// runPipeline executes one tracking step. The steps below are ordered so
// that high-confidence, motion-consistent associations happen before any
// fallback (appearance re-identification, low-confidence association,
// unconfirmed matching), and new tracks are only spawned from detections
// that nothing else claimed.
func (t *Tracker[T]) runPipeline(dets []T) ([]Track[T], []Track[T]) {
	detsHigh, detsLow := t.prepareDetections(dets)
	unconfirmed, matchable := t.prepareTracks()

	var (
		activated   []*track[T]
		reactivated []*track[T]
		lostNow     []*track[T]
		removedNow  []*track[T]
	)

	// Appearance-driven re-identification of Reidable tracks before the
	// motion-based association pass. Anything promoted here joins the
	// matchable pool and is eligible for a fresh detection below.
	matchable, removedNow = t.reidentificationStep(matchable, removedNow)

	pureDetIDs := t.computePureDetectionIDs(detsHigh)

	// First association: confirmed tracks against high-confidence detections,
	// using a fused IoU (+ optional embedding) cost.
	highAct, highReact, unmatchedTracks, unmatchedDetsHigh := t.firstAssociation(
		matchable, detsHigh, pureDetIDs)
	activated = append(activated, highAct...)
	reactivated = append(reactivated, highReact...)

	// Second association: remaining Tracked tracks against low-confidence
	// detections. Tracks that survive an occluder often resurface as a
	// low-score detection, so this pass is essential for continuity.
	lowAct, lowReact, lostTracks := t.secondAssociation(unmatchedTracks, detsLow)
	activated = append(activated, lowAct...)
	reactivated = append(reactivated, lowReact...)
	lostNow = append(lostNow, lostTracks...)

	// Unconfirmed tracks only survive if they match a high-confidence
	// detection on their second frame; otherwise they are dropped to avoid
	// promoting spurious single-frame detections.
	unconfAct, unconfRemoved, detRemain := t.handleUnconfirmed(unconfirmed, unmatchedDetsHigh)
	activated = append(activated, unconfAct...)
	removedNow = append(removedNow, unconfRemoved...)

	newTracks := t.initNewTracks(detRemain, pureDetIDs)
	activated = append(activated, newTracks...)

	output, removedReturned := t.updateState(removedNow, activated, reactivated, lostNow)

	outputIfaces := asIfaces(output)
	removedIfaces := asIfaces(removedReturned)
	slices.SortFunc(outputIfaces, CmpTrackID[T])
	slices.SortFunc(removedIfaces, CmpTrackID[T])
	return outputIfaces, removedIfaces
}

func asIfaces[T Detection](tracks []*track[T]) []Track[T] {
	out := make([]Track[T], len(tracks))
	for i, tr := range tracks {
		out[i] = tr
	}
	return out
}

// prepareDetections converts raw detections into detInfo records and
// partitions them into "high" (score > DetHighThresh) and "low" pools.
// Embeddings are stripped from the low pool: low-confidence matches should
// not feed the appearance EMA, which would otherwise be polluted by noisy
// observations of partially-occluded objects.
func (t *Tracker[T]) prepareDetections(dets []T) (high, low []detInfo[T]) {
	n := len(dets)
	if n == 0 {
		return nil, nil
	}

	// Visibility ratios are computed over *all* detections before splitting,
	// so a box partly hidden by a below-threshold detection is still flagged
	// as occluded.
	info := make([]detInfo[T], n)
	boxes := make([][4]float64, n)
	for i, d := range dets {
		xyxy := d.GetXYXY()
		info[i] = detInfo[T]{
			det:   d,
			xywh:  utils.XYXY2XYWH(xyxy),
			conf:  d.GetScore(),
			class: getClass(d),
			detID: getDetID(d, i),
		}
		boxes[i] = info[i].xywh

		if emb, ok := any(d).(DetectionWithEmbedding); ok {
			info[i].emb = emb.GetEmbedding()
		}
	}

	if t.WithEmbReactivation {
		vrs := utils.BoxVisibilityRatiosBatch(boxes)
		for i := range info {
			info[i].overlap = vrs[i]
		}
	}

	for i := range info {
		c := info[i].conf
		switch {
		case c > t.DetHighThresh:
			high = append(high, info[i])
		case c > t.DetLowThresh:
			lo := info[i]
			lo.emb = nil
			low = append(low, lo)
		}
	}
	return high, low
}

func getClass[T Detection](d T) int {
	if c, ok := any(d).(DetectionWithClass); ok {
		return c.GetClass()
	}
	return 0
}

func getDetID[T Detection](d T, fallback int) int {
	if id, ok := any(d).(DetectionWithIndex); ok {
		return id.GetDetID()
	}
	return fallback
}

// prepareTracks partitions the active and lost pools into the unconfirmed
// and matchable working lists, then advances every matchable track one step
// with the Kalman filter. Tracks that missed the previous frame (state !=
// StateTracked) have their height velocity reset to zero: without an
// observation, a stale vh would keep shrinking or growing the box
// unboundedly while the track is coasting.
func (t *Tracker[T]) prepareTracks() (unconfirmed, matchable []*track[T]) {
	for _, id := range sortedIDs(t.active) {
		tr := t.active[id]
		if tr.activated {
			matchable = append(matchable, tr)
		} else {
			unconfirmed = append(unconfirmed, tr)
		}
	}
	for _, id := range sortedIDs(t.lost) {
		matchable = append(matchable, t.lost[id])
	}

	for _, tr := range matchable {
		if tr.state != StateTracked {
			tr.mean[7] = 0
		}
	}

	if len(matchable) > 0 {
		means := make([][8]float64, len(matchable))
		covs := make([][64]float64, len(matchable))
		for i, tr := range matchable {
			means[i] = tr.mean
			covs[i] = tr.cov
		}
		t.kf.MultiPredict(means, covs)
		for i, tr := range matchable {
			tr.mean = means[i]
			tr.cov = covs[i]
		}
	}
	return unconfirmed, matchable
}

// reidentificationStep matches long-lost (Reidable) tracks against currently
// active tracks by appearance, under the hypothesis that an "active" track
// is really a new id for a previously lost object. When a match passes the
// EmbReIDThresh gate, the Reidable track absorbs the active track's motion
// state, the active id is Removed, and the revived Reidable track is
// promoted back into the matchable pool so first association can still
// bind a detection to it this frame.
//
// The age filter (reidable id < active id) guarantees that the older track
// always wins — without it, two short-lived ids could bounce their state
// back and forth and destabilize the output.
func (t *Tracker[T]) reidentificationStep(matchable, removed []*track[T]) (
	matchableOut, removedOut []*track[T],
) {
	if !t.WithReID || !t.WithEmbReactivation || len(t.reidable) == 0 {
		return matchable, removed
	}

	var reidTracks []*track[T]
	for _, tr := range t.reidable {
		reidTracks = append(reidTracks, tr)
	}
	// Only established active tracks participate — requiring pureCnt ≥
	// MinPureCnt ensures that the active side has a trustworthy embedding
	// history before we consider merging it away.
	var actualTracks []*track[T]
	for _, tr := range t.active {
		if tr.activated && tr.pureCnt >= t.MinPureCnt {
			actualTracks = append(actualTracks, tr)
		}
	}
	if len(reidTracks) == 0 || len(actualTracks) == 0 {
		return matchable, removed
	}

	reidEmbs := make([][]float64, len(reidTracks))
	for i, tr := range reidTracks {
		reidEmbs[i] = tr.pureEmb
	}
	actualEmbs := make([][]float64, len(actualTracks))
	for i, tr := range actualTracks {
		actualEmbs[i] = tr.pureEmb
	}
	dist := utils.EmbeddingDistance(reidEmbs, actualEmbs)

	matches, _, _ := t.lapSolver.Solve(
		len(reidTracks), len(actualTracks), dist, t.EmbReIDThresh)

	removeSet := make(map[uint]struct{}, len(matches))
	var promoted []*track[T]
	for _, m := range matches {
		rIdx, aIdx := m[0], m[1]
		reidT := reidTracks[rIdx]
		actualT := actualTracks[aIdx]
		if reidT.id >= actualT.id {
			continue
		}
		t.mergeInto(reidT, actualT)
		removeSet[actualT.id] = struct{}{}
		t.reidable, t.active = promoteReidable(t.reidable, t.active, reidT)
		promoted = append(promoted, reidT)
	}

	if len(removeSet) == 0 && len(promoted) == 0 {
		return matchable, removed
	}

	filtered := matchable[:0]
	for _, tr := range matchable {
		if _, dead := removeSet[tr.id]; !dead {
			filtered = append(filtered, tr)
		}
	}
	matchable = filtered
	for id := range removeSet {
		if dead, ok := t.active[id]; ok {
			dead.state = StateRemoved
			removed = append(removed, dead)
			delete(t.active, id)
		}
	}
	matchable = append(matchable, promoted...)
	return matchable, removed
}

// mergeInto folds the source track (an active id being retired) into the
// destination (an older Reidable track that is being revived). The Kalman
// state and the latest detection-side fields are copied wholesale so that
// the revived track picks up smoothly from where the source was; the stored
// embeddings are EMA-blended so that the revived track benefits from the
// source's recent appearance observations without discarding its own.
//
// The destination keeps its identity (id, startFrame, pureCnt) — that is
// the whole point of re-identification.
func (t *Tracker[T]) mergeInto(dst, src *track[T]) {
	dst.mean = src.mean
	dst.cov = src.cov
	dst.frameID = src.frameID
	dst.state = src.state
	dst.activated = src.activated

	if src.emb != nil && dst.emb != nil {
		t.ema.UpdateRow(dst.emb, dst.emb, src.emb, -1)
	}
	if src.pureEmb != nil && dst.pureEmb != nil {
		t.ema.UpdateRow(dst.pureEmb, dst.pureEmb, src.pureEmb, -1)
	}

	dst.conf = src.conf
	dst.class = src.class
	dst.detID = src.detID
	dst.detection = src.detection
}

func promoteReidable[T Detection](reidPool, activePool map[uint]*track[T], tr *track[T]) (map[uint]*track[T], map[uint]*track[T]) {
	delete(reidPool, tr.id)
	activePool[tr.id] = tr
	return reidPool, activePool
}

// computePureDetectionIDs returns the detection ids that qualify as "pure",
// i.e. clearly unoccluded and large enough to provide a reliable appearance
// signal. A detection is pure when its visibility ratio exceeds VRThresh
// and, when the image size is known, covers at least 0.1 % of the frame.
// Only pure detections are allowed to update the pureEmb that drives
// long-range re-identification.
func (t *Tracker[T]) computePureDetectionIDs(detsHigh []detInfo[T]) map[int]struct{} {
	if !t.WithEmbReactivation {
		return nil
	}
	out := make(map[int]struct{}, len(detsHigh))
	imgArea := float64(t.imageW * t.imageH)
	for _, d := range detsHigh {
		vis := 1 - d.overlap
		if vis <= t.VRThresh {
			continue
		}
		if imgArea > 0 {
			area := vis * (d.xywh[2] * d.xywh[3]) / imgArea
			if area < 0.001 {
				continue
			}
		}
		out[d.detID] = struct{}{}
	}
	return out
}

// firstAssociation matches confirmed and lost tracks against high-confidence
// detections. When WithReID is enabled and every participant carries an
// embedding, the cost matrix fuses motion (IoU-based) and appearance
// (cosine) costs with α = IoUReIDAlpha; otherwise it is pure motion cost.
// Matches that exceed MatchDetHighThresh are rejected.
func (t *Tracker[T]) firstAssociation(
	tracks []*track[T], dets []detInfo[T], pureIDs map[int]struct{},
) (activated, reactivated, unmatchedTracks []*track[T], unmatchedDets []detInfo[T]) {
	if len(tracks) == 0 || len(dets) == 0 {
		return nil, nil, tracks, dets
	}

	tBoxes := make([][4]float64, len(tracks))
	for i, tr := range tracks {
		tBoxes[i] = [4]float64{tr.mean[0], tr.mean[1], tr.mean[2], tr.mean[3]}
	}
	dBoxes := make([][4]float64, len(dets))
	dConfs := make([]float64, len(dets))
	for i, d := range dets {
		dBoxes[i] = d.xywh
		dConfs[i] = d.conf
	}

	t.iouBuf = utils.IoUBatchXYWH(t.iouBuf, tBoxes, dBoxes)
	oneMinusInPlace(t.iouBuf)
	iouDists := utils.FuseScore(t.iouBuf, t.iouBuf, dConfs)
	t.iouBuf = iouDists

	if t.WithReID && hasEmb(dets) && hasTrackEmb(tracks) {
		tEmbs := make([][]float64, len(tracks))
		dEmbs := make([][]float64, len(dets))
		for i, tr := range tracks {
			tEmbs[i] = tr.emb
		}
		for i, d := range dets {
			dEmbs[i] = d.emb
		}
		embDists := utils.EmbeddingDistance(tEmbs, dEmbs)
		// Cross-gating: either signal alone is not enough to carry a match.
		// If appearance is already poor, ignore the motion vote; if motion
		// is already poor, ignore the appearance vote; and drop any match
		// whose appearance cost exceeds EmbThresh outright.
		for i := range embDists {
			for j := range embDists[i] {
				if embDists[i][j] > t.IoUEmbThresh {
					iouDists[i][j] = 1.0
				}
				if embDists[i][j] > t.EmbThresh {
					embDists[i][j] = 1.0
				}
				if iouDists[i][j] > t.EmbIoUThresh {
					embDists[i][j] = 1.0
				}
			}
		}
		// Fold the fused cost back into iouDists (same storage) — each
		// iouDists[i][j] is read once before being overwritten, so the
		// aliasing is safe.
		alpha := t.IoUReIDAlpha
		for i := range tracks {
			row := iouDists[i]
			erow := embDists[i]
			for j := range dets {
				row[j] = alpha*row[j] + (1-alpha)*erow[j]
			}
		}
	}

	matches, uT, uD := t.lapSolver.Solve(
		len(tracks), len(dets), iouDists, t.MatchDetHighThresh)

	// Matches of Tracked tracks continue forward through a Kalman update;
	// matches of Lost tracks go through the reactivation path instead.
	var updT, reactT []*track[T]
	var updD, reactD []detInfo[T]
	var updPureMask []bool
	for _, m := range matches {
		trIdx, dIdx := m[0], m[1]
		tr := tracks[trIdx]
		d := dets[dIdx]
		if tr.state == StateTracked {
			updT = append(updT, tr)
			updD = append(updD, d)
			if pureIDs != nil {
				_, isPure := pureIDs[d.detID]
				updPureMask = append(updPureMask, isPure)
			}
		} else {
			reactT = append(reactT, tr)
			reactD = append(reactD, d)
		}
	}
	t.processUpdate(updT, updD, updPureMask)
	t.processReactivate(reactT, reactD)
	activated = updT
	reactivated = reactT

	unmatchedTracks = make([]*track[T], 0, len(uT))
	for _, idx := range uT {
		unmatchedTracks = append(unmatchedTracks, tracks[idx])
	}
	unmatchedDets = make([]detInfo[T], 0, len(uD))
	for _, idx := range uD {
		unmatchedDets = append(unmatchedDets, dets[idx])
	}
	return activated, reactivated, unmatchedTracks, unmatchedDets
}

func hasEmb[T Detection](dets []detInfo[T]) bool {
	for i := range dets {
		if dets[i].emb == nil {
			return false
		}
	}
	return len(dets) > 0
}

func hasTrackEmb[T Detection](tracks []*track[T]) bool {
	for _, tr := range tracks {
		if tr.emb == nil {
			return false
		}
	}
	return len(tracks) > 0
}

// processUpdate applies the Kalman correction for a batch of Tracked tracks
// that matched detections, then blends the appearance embedding and, for
// detections that passed the pure filter, the pure embedding as well.
//
// NSA is deliberately disabled on the update step: empirically, scaling R
// by (1 - conf)² inside a tight gate over-trusts high-confidence detections
// and causes the covariance to collapse, which in turn causes the track to
// refuse future associations when the object appearance drifts.
func (t *Tracker[T]) processUpdate(tracks []*track[T], dets []detInfo[T], pureMask []bool) {
	if len(tracks) == 0 {
		return
	}
	means := make([][8]float64, len(tracks))
	covs := make([][64]float64, len(tracks))
	meas := make([][4]float64, len(tracks))
	for i, tr := range tracks {
		means[i] = tr.mean
		covs[i] = tr.cov
		meas[i] = dets[i].xywh
	}
	t.kf.MultiUpdate(means, covs, meas, nil)
	for i, tr := range tracks {
		tr.mean = means[i]
		tr.cov = covs[i]
		tr.detection = dets[i].det
		tr.class = dets[i].class
		tr.detID = dets[i].detID
		tr.conf = dets[i].conf
		tr.frameID = t.frameID
		tr.state = StateTracked
		tr.activated = true

		if tr.emb != nil && dets[i].emb != nil {
			t.ema.UpdateRow(tr.emb, tr.emb, dets[i].emb, dets[i].conf)
		}
	}

	if pureMask == nil {
		return
	}
	for i, tr := range tracks {
		if !pureMask[i] {
			continue
		}
		if dets[i].emb == nil {
			continue
		}
		if tr.pureCnt == 0 {
			// The very first pure observation is stored raw (normalized)
			// rather than EMA-blended — without an existing value to blend
			// against, the EMA would effectively down-scale it.
			if tr.pureEmb == nil {
				tr.pureEmb = make([]float64, len(dets[i].emb))
			}
			copyNormalized(tr.pureEmb, dets[i].emb)
		} else {
			t.ema.UpdateRow(tr.pureEmb, tr.pureEmb, dets[i].emb, dets[i].conf)
		}
		tr.pureCnt++
	}
}

// copyNormalized writes the L2-normalized src into dst, rounding each
// element through float32. The float32 rounding matches the precision of
// the upstream embedding pipeline; without it, long EMA chains drift far
// enough to cross the re-identification threshold.
func copyNormalized(dst, src []float64) {
	var n float64
	for _, x := range src {
		n += x * x
	}
	if n == 0 {
		copy(dst, src)
		return
	}
	inv := 1.0 / math.Sqrt(n)
	for i, x := range src {
		dst[i] = float64(float32(x * inv))
	}
}

// processReactivate is the Lost → Tracked transition: same Kalman and EMA
// machinery as processUpdate, but without the pureEmb update, since a
// reactivation is not a pure observation by definition.
func (t *Tracker[T]) processReactivate(tracks []*track[T], dets []detInfo[T]) {
	if len(tracks) == 0 {
		return
	}
	means := make([][8]float64, len(tracks))
	covs := make([][64]float64, len(tracks))
	meas := make([][4]float64, len(tracks))
	for i, tr := range tracks {
		means[i] = tr.mean
		covs[i] = tr.cov
		meas[i] = dets[i].xywh
	}
	t.kf.MultiUpdate(means, covs, meas, nil)
	for i, tr := range tracks {
		tr.mean = means[i]
		tr.cov = covs[i]
		tr.detection = dets[i].det
		tr.class = dets[i].class
		tr.detID = dets[i].detID
		tr.conf = dets[i].conf
		tr.frameID = t.frameID
		tr.state = StateTracked
		tr.activated = true
		if tr.emb != nil && dets[i].emb != nil {
			t.ema.UpdateRow(tr.emb, tr.emb, dets[i].emb, dets[i].conf)
		}
	}
}

// secondAssociation attempts to bind unmatched Tracked tracks to
// low-confidence detections using pure IoU cost. Lost tracks are not
// admitted to this pass: a low-confidence detection is too weak a signal
// to reactivate a track that has already been absent for a frame.
func (t *Tracker[T]) secondAssociation(unmatched []*track[T], detsLow []detInfo[T]) (
	activated, reactivated, lost []*track[T],
) {
	var remain []*track[T]
	for _, tr := range unmatched {
		if tr.state == StateTracked {
			remain = append(remain, tr)
		}
	}
	if len(remain) == 0 || len(detsLow) == 0 {
		for _, tr := range remain {
			tr.state = StateLost
			lost = append(lost, tr)
		}
		return nil, nil, lost
	}

	tBoxes := make([][4]float64, len(remain))
	for i, tr := range remain {
		tBoxes[i] = [4]float64{tr.mean[0], tr.mean[1], tr.mean[2], tr.mean[3]}
	}
	dBoxes := make([][4]float64, len(detsLow))
	for i, d := range detsLow {
		dBoxes[i] = d.xywh
	}
	t.iouBuf = utils.IoUBatchXYWH(t.iouBuf, tBoxes, dBoxes)
	oneMinusInPlace(t.iouBuf)

	matches, uT, _ := t.lapSolver.Solve(
		len(remain), len(detsLow), t.iouBuf, t.MatchDetLowThresh)

	var updT, reactT []*track[T]
	var updD, reactD []detInfo[T]
	for _, m := range matches {
		tr := remain[m[0]]
		d := detsLow[m[1]]
		if tr.state == StateTracked {
			updT = append(updT, tr)
			updD = append(updD, d)
		} else {
			reactT = append(reactT, tr)
			reactD = append(reactD, d)
		}
	}
	t.processUpdate(updT, updD, nil)
	t.processReactivate(reactT, reactD)
	activated = updT
	reactivated = reactT

	for _, idx := range uT {
		tr := remain[idx]
		tr.state = StateLost
		lost = append(lost, tr)
	}
	return activated, reactivated, lost
}

// handleUnconfirmed attempts to confirm unconfirmed tracks (those born on
// a previous frame that have not yet been matched a second time) against
// the high-confidence detections the two association passes left behind.
// Any unconfirmed track that fails to match here is Removed outright —
// this is the mechanism that prevents one-frame spurious detections from
// becoming persistent tracks.
func (t *Tracker[T]) handleUnconfirmed(unconfirmed []*track[T], unmatchedHigh []detInfo[T]) (
	activated, removed []*track[T], remain []detInfo[T],
) {
	if len(unconfirmed) == 0 {
		return nil, nil, unmatchedHigh
	}
	if len(unmatchedHigh) == 0 {
		for _, tr := range unconfirmed {
			tr.state = StateRemoved
			removed = append(removed, tr)
		}
		return nil, removed, nil
	}

	tBoxes := make([][4]float64, len(unconfirmed))
	for i, tr := range unconfirmed {
		tBoxes[i] = [4]float64{tr.mean[0], tr.mean[1], tr.mean[2], tr.mean[3]}
	}
	dBoxes := make([][4]float64, len(unmatchedHigh))
	dConfs := make([]float64, len(unmatchedHigh))
	for i, d := range unmatchedHigh {
		dBoxes[i] = d.xywh
		dConfs[i] = d.conf
	}
	t.iouBuf = utils.IoUBatchXYWH(t.iouBuf, tBoxes, dBoxes)
	oneMinusInPlace(t.iouBuf)
	t.iouBuf = utils.FuseScore(t.iouBuf, t.iouBuf, dConfs)

	matches, uT, uD := t.lapSolver.Solve(
		len(unconfirmed), len(unmatchedHigh), t.iouBuf, t.UnconfMatchThresh)

	var updT []*track[T]
	var updD []detInfo[T]
	for _, m := range matches {
		updT = append(updT, unconfirmed[m[0]])
		updD = append(updD, unmatchedHigh[m[1]])
	}
	t.processUpdate(updT, updD, nil)
	activated = updT

	for _, idx := range uT {
		tr := unconfirmed[idx]
		tr.state = StateRemoved
		removed = append(removed, tr)
	}
	remain = make([]detInfo[T], 0, len(uD))
	for _, idx := range uD {
		remain = append(remain, unmatchedHigh[idx])
	}
	return activated, removed, remain
}

// initNewTracks spawns a new track for every remaining high-confidence
// detection whose score exceeds TrackNewThresh. Tracks born on the very
// first frame are marked activated immediately; on later frames they start
// as unconfirmed and must survive handleUnconfirmed on the next frame.
func (t *Tracker[T]) initNewTracks(dets []detInfo[T], pureIDs map[int]struct{}) []*track[T] {
	var filtered []detInfo[T]
	for _, d := range dets {
		if d.conf >= t.TrackNewThresh {
			filtered = append(filtered, d)
		}
	}
	if len(filtered) == 0 {
		return nil
	}

	boxes := make([][4]float64, len(filtered))
	for i, d := range filtered {
		boxes[i] = d.xywh
	}
	means := make([][8]float64, len(filtered))
	covs := make([][64]float64, len(filtered))
	t.kf.MultiInitiate(boxes, means, covs)

	out := make([]*track[T], len(filtered))
	for i, d := range filtered {
		tr := &track[T]{
			id:               t.nextID,
			state:            StateTracked,
			activated:        t.frameID == 1,
			detection:        d.det,
			class:            d.class,
			detID:            d.detID,
			conf:             d.conf,
			mean:             means[i],
			cov:              covs[i],
			startFrame:       t.frameID,
			firstDetectionID: d.detID,
			frameID:          t.frameID,
		}
		t.nextID++
		if d.emb != nil {
			// Birth stores the raw (un-normalized) detection vector in both
			// slots. Normalization is delayed until the first pure update
			// post-birth, so that the EMA chain starts from a known value.
			tr.emb = slices.Clone(d.emb)
			if pureIDs != nil {
				tr.pureEmb = make([]float64, len(d.emb))
				if _, ok := pureIDs[d.detID]; ok {
					copy(tr.pureEmb, d.emb)
					tr.pureCnt = 1
				}
			}
		}
		out[i] = tr
	}
	return out
}

// updateState reconciles the active / lost / reidable pools after the
// per-frame matching decisions have been made and returns the tracks to
// surface to the caller (activated Tracked tracks and tracks that became
// Removed this frame).
func (t *Tracker[T]) updateState(
	removed, activated, reactivated, lost []*track[T],
) (output, removedOut []*track[T]) {

	// Age out Lost tracks. Those that accumulated enough pure observations
	// transition to Reidable; the rest are Removed.
	var newReidable []*track[T]
	var newRemoved []*track[T]
	for id, tr := range t.lost {
		if t.frameID-tr.frameID <= t.MaxFramesLost {
			continue
		}
		delete(t.lost, id)
		if t.WithEmbReactivation && tr.pureCnt >= t.MinPureCnt {
			tr.state = StateReidable
			tr.frameID = t.frameID
			newReidable = append(newReidable, tr)
		} else {
			tr.state = StateRemoved
			newRemoved = append(newRemoved, tr)
		}
	}
	// Age out Reidable tracks that ran past their window.
	if t.WithEmbReactivation {
		for id, tr := range t.reidable {
			if t.frameID-tr.frameID > t.MaxFramesReidable {
				tr.state = StateRemoved
				newRemoved = append(newRemoved, tr)
				delete(t.reidable, id)
			}
		}
	}

	// Evict active entries that no longer hold state == Tracked, then
	// fold in the freshly activated and reactivated tracks.
	for id, tr := range t.active {
		if tr.state != StateTracked {
			delete(t.active, id)
		}
	}
	for _, tr := range activated {
		t.active[tr.id] = tr
	}
	for _, tr := range reactivated {
		t.active[tr.id] = tr
	}

	// Admit newly-Lost tracks into the lost pool, then scrub anything that
	// re-entered active or was promoted to reidable this frame.
	for _, tr := range lost {
		if _, inActive := t.active[tr.id]; inActive {
			continue
		}
		t.lost[tr.id] = tr
	}
	for id := range t.active {
		delete(t.lost, id)
	}
	for _, tr := range newReidable {
		delete(t.lost, tr.id)
	}

	for id := range t.active {
		delete(t.reidable, id)
	}
	for _, tr := range newReidable {
		t.reidable[tr.id] = tr
	}

	// Removed tracks are surfaced exactly once and never stored.
	removedOut = append(removedOut, removed...)
	removedOut = append(removedOut, newRemoved...)

	t.removeDuplicates()

	for _, tr := range t.active {
		if tr.activated {
			output = append(output, tr)
		}
	}
	return output, removedOut
}

// removeDuplicates resolves active/lost track pairs whose boxes overlap
// more than 1 - RemoveDuplicateThresh. The younger track (by age relative
// to its startFrame) is dropped, so when a new id appears near an existing
// older track — almost always a spurious duplicate — the long-lived id
// survives. Ties break toward keeping the active side.
func (t *Tracker[T]) removeDuplicates() {
	if len(t.active) == 0 || len(t.lost) == 0 {
		return
	}
	activeList := make([]*track[T], 0, len(t.active))
	lostList := make([]*track[T], 0, len(t.lost))
	for _, tr := range t.active {
		activeList = append(activeList, tr)
	}
	for _, tr := range t.lost {
		lostList = append(lostList, tr)
	}
	aBoxes := make([][4]float64, len(activeList))
	for i, tr := range activeList {
		aBoxes[i] = [4]float64{tr.mean[0], tr.mean[1], tr.mean[2], tr.mean[3]}
	}
	bBoxes := make([][4]float64, len(lostList))
	for i, tr := range lostList {
		bBoxes[i] = [4]float64{tr.mean[0], tr.mean[1], tr.mean[2], tr.mean[3]}
	}
	t.iouBuf = utils.IoUBatchXYWH(t.iouBuf, aBoxes, bBoxes)

	threshIoU := 1 - t.RemoveDuplicateThresh
	dropA := make(map[uint]struct{})
	dropB := make(map[uint]struct{})
	for i, row := range t.iouBuf {
		for j, v := range row {
			if v <= threshIoU {
				continue
			}
			a := activeList[i]
			b := lostList[j]
			aAge := a.frameID - a.startFrame
			bAge := b.frameID - b.startFrame
			if aAge < bAge {
				dropA[a.id] = struct{}{}
			}
			if bAge <= aAge {
				dropB[b.id] = struct{}{}
			}
		}
	}
	for id := range dropA {
		delete(t.active, id)
	}
	for id := range dropB {
		delete(t.lost, id)
	}
}

// sortedIDs returns the keys of a track pool in ascending order. Iteration
// order matters because it determines the row order of the cost matrix,
// and the assignment solver is sensitive to ordering when several matches
// are equally good.
func sortedIDs[T Detection](pool map[uint]*track[T]) []uint {
	out := make([]uint, 0, len(pool))
	for id := range pool {
		out = append(out, id)
	}
	slices.Sort(out)
	return out
}

// oneMinusInPlace replaces each cell of m with 1 - v.
func oneMinusInPlace(m [][]float64) {
	for _, row := range m {
		for j, v := range row {
			row[j] = 1 - v
		}
	}
}
