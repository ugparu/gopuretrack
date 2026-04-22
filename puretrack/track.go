package puretrack

// Detection is the minimal contract every input must satisfy: an axis-aligned
// bounding box in (x1, y1, x2, y2) pixel coordinates and a confidence score.
type Detection interface {
	GetXYXY() [4]float64
	GetScore() float64
}

// DetectionWithEmbedding supplies an appearance embedding for ReID matching.
// A Tracker configured with WithReID=true expects every detection to satisfy
// this interface.
type DetectionWithEmbedding interface {
	Detection
	GetEmbedding() []float64
}

// DetectionWithClass carries a class label. Detections that do not implement
// it are treated as class 0.
type DetectionWithClass interface {
	Detection
	GetClass() int
}

// DetectionWithIndex carries a caller-assigned detection id that is copied
// onto the matched track. Detections that do not implement it receive a
// per-frame sequential index instead.
type DetectionWithIndex interface {
	Detection
	GetDetID() int
}

// TrackState is the lifecycle state of a track.
//
//	New      — just spawned; not yet confirmed by a second match
//	Tracked  — confirmed and observed on the current frame
//	Lost     — missed this frame but still within MaxFramesLost
//	Reidable — past MaxFramesLost but still within MaxFramesReidable;
//	           eligible for appearance-based reactivation if the track has
//	           accumulated at least MinPureCnt unoccluded observations
//	Removed  — permanently retired; never returned again
type TrackState uint8

// The lifecycle states of a track, in the order they are entered. See
// TrackState for a description of each value.
const (
	StateNew TrackState = iota
	StateTracked
	StateLost
	StateReidable
	StateRemoved
)

// Track is the minimal external view of a tracked object. Callers that need
// additional per-track fields can type-assert to TrackExt.
type Track[T Detection] interface {
	GetDetection() T
	GetID() uint
	GetXYXY() [4]float64
}

// TrackExt exposes confidence, class, detection id and lifecycle state in
// addition to the Track interface.
type TrackExt[T Detection] interface {
	Track[T]
	GetScore() float64
	GetClass() int
	GetDetID() int
	GetState() TrackState
}

type track[T Detection] struct {
	id        uint
	state     TrackState
	activated bool

	detection T
	class     int
	detID     int
	conf      float64

	// Kalman state in (x, y, w, h, vx, vy, vw, vh) and its 8×8 covariance
	// stored row-major.
	mean [8]float64
	cov  [64]float64

	// emb is the running appearance embedding, updated with every match.
	// nil when the tracker runs without ReID.
	emb []float64
	// pureEmb is an appearance embedding restricted to unoccluded
	// observations (visibility ratio ≥ VRThresh and, when the image size
	// is known, area ≥ 0.1 % of the frame). It is more robust to occluders
	// and is what drives re-identification of stale tracks.
	pureEmb []float64
	pureCnt uint

	startFrame       uint
	firstDetectionID int
	frameID          uint
}

func (t *track[T]) GetDetection() T      { return t.detection }
func (t *track[T]) GetID() uint          { return t.id }
func (t *track[T]) GetScore() float64    { return t.conf }
func (t *track[T]) GetClass() int        { return t.class }
func (t *track[T]) GetDetID() int        { return t.detID }
func (t *track[T]) GetState() TrackState { return t.state }

// GetXYXY returns the current bounding box in (x1, y1, x2, y2) pixel
// coordinates, reconstructed from the filter's center-plus-size state.
func (t *track[T]) GetXYXY() [4]float64 {
	x, y, w, h := t.mean[0], t.mean[1], t.mean[2], t.mean[3]
	return [4]float64{x - w/2, y - h/2, x + w/2, y + h/2}
}

// CmpTrackID orders tracks by ascending id. Since ids are issued
// monotonically, this is also chronological birth order.
func CmpTrackID[T Detection](t1, t2 Track[T]) int {
	if t1.GetID() < t2.GetID() {
		return -1
	}
	if t1.GetID() > t2.GetID() {
		return 1
	}
	return 0
}
