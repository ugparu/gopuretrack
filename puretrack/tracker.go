// Package puretrack is a multi-object tracker that associates detections
// into persistent tracks across video frames. It combines motion modeling
// (a constant-velocity Kalman filter on bounding boxes) with optional
// appearance-based re-identification, a two-stage cascade of high- and
// low-confidence associations, and an explicit lifecycle for lost and
// re-identifiable tracks.
//
// The public entry point is Tracker[T], generic over the caller's
// detection type. A minimal detection satisfies the Detection interface;
// optional sub-interfaces (DetectionWithEmbedding, DetectionWithClass,
// DetectionWithIndex) are detected via type assertion per call.
package puretrack

import (
	"github.com/ugparu/gopuretrack/embedding"
	"github.com/ugparu/gopuretrack/kalman"
	"github.com/ugparu/gopuretrack/utils/lap"
)

// Config is the full set of tracker knobs. Any zero-valued numeric field is
// transparently replaced with the corresponding BaseConfig value, so a
// short literal is a valid config.
type Config struct {
	// Detection gating.

	// TrackNewThresh is the minimum confidence a high-conf detection must
	// exceed to spawn a new track.
	TrackNewThresh float64
	// DetHighThresh splits detections into the "high" pool, which drives the
	// first-association pass and new-track initialization.
	DetHighThresh float64
	// DetLowThresh is the floor for the "low" pool used by the second
	// association pass; anything below is discarded.
	DetLowThresh float64

	// Association thresholds. All three are upper bounds on association
	// cost, so matches with cost above the threshold are rejected.
	MatchDetHighThresh float64
	MatchDetLowThresh  float64
	UnconfMatchThresh  float64

	// RemoveDuplicateThresh is the maximum 1-IoU distance between an active
	// and a lost track before one is treated as a duplicate.
	RemoveDuplicateThresh float64

	// MaxFramesLost is how many frames a track may be missed before it
	// transitions from Lost to Reidable (or Removed, if ineligible).
	MaxFramesLost uint
	// MaxFramesReidable is how many frames a Reidable track is kept alive
	// for appearance-based re-identification before it is Removed.
	MaxFramesReidable uint

	// ReID gating (consulted only when WithReID=true).

	// IoUEmbThresh zeroes out motion cost for track/det pairs whose
	// appearance distance is already above this threshold.
	IoUEmbThresh float64
	// EmbIoUThresh zeroes out appearance cost when the motion cost is
	// already above this threshold — the two thresholds are deliberately
	// symmetric so a single bad signal cannot carry a match.
	EmbIoUThresh float64
	// EmbThresh is the maximum appearance distance admitted into the fused
	// cost matrix.
	EmbThresh float64
	// VRThresh is the minimum visibility ratio (1 − max overlap with any
	// other detection) for a detection to be considered "pure".
	VRThresh float64
	// EmbReIDThresh is the cost ceiling used by the appearance-only
	// re-identification pass.
	EmbReIDThresh float64
	// IoUReIDAlpha is the weight of motion cost in the fused
	// motion+appearance cost matrix. (1 − α) weights the appearance term.
	IoUReIDAlpha float64

	// WithReID enables appearance-based matching. It requires every
	// detection to implement DetectionWithEmbedding.
	WithReID bool
	// WithEmbReactivation enables the Reidable pool and the pure-embedding
	// re-identification pass that resurrects long-lost tracks.
	WithEmbReactivation bool
	// MinPureCnt is the minimum number of pure observations a track must
	// accumulate before it is eligible for Reidable status.
	MinPureCnt uint
	// EmbMode selects the appearance update rule (EMA or last-value).
	EmbMode embedding.Mode
	// EmbEMAAlpha is the retention weight of the previous embedding when
	// EmbMode is ModeEMA; the new observation contributes (1 − α) · conf.
	EmbEMAAlpha float64

	// Kalman filter weights. All three std-weight fields are fractions of
	// the current bbox width/height; the init factors widen the birth
	// covariance for a more tolerant first association.
	KFStdWeightProcessPos  float64
	KFStdWeightProcessVel  float64
	KFStdWeightMeasurement float64
	KFInitPosCovFactor     float64
	KFInitVelCovFactor     float64
}

// BaseConfig is the default configuration. The frame-count limits assume
// a 30 fps source: MaxFramesLost ≈ 1.2 s and MaxFramesReidable ≈ 5 s.
var BaseConfig = Config{
	TrackNewThresh:         0.6,
	DetHighThresh:          0.6,
	DetLowThresh:           0.1,
	MatchDetHighThresh:     0.9,
	MatchDetLowThresh:      0.5,
	UnconfMatchThresh:      0.7,
	RemoveDuplicateThresh:  0.15,
	MaxFramesLost:          36,
	MaxFramesReidable:      150,
	IoUEmbThresh:           0.38,
	EmbIoUThresh:           0.38,
	EmbThresh:              0.5,
	VRThresh:               0.3,
	EmbReIDThresh:          0.15,
	IoUReIDAlpha:           0.9,
	WithReID:               false,
	WithEmbReactivation:    true,
	MinPureCnt:             15,
	EmbMode:                embedding.ModeEMA,
	EmbEMAAlpha:            0.9,
	KFStdWeightProcessPos:  0.05,
	KFStdWeightProcessVel:  0.00625,
	KFStdWeightMeasurement: 0.05,
	KFInitPosCovFactor:     2.0,
	KFInitVelCovFactor:     10.0,
}

// fillDefaults replaces zero-valued numeric fields with BaseConfig values.
// Bool fields are left untouched so that callers can explicitly disable
// features that are on by default.
func (c *Config) fillDefaults() {
	if c.TrackNewThresh == 0 {
		c.TrackNewThresh = BaseConfig.TrackNewThresh
	}
	if c.DetHighThresh == 0 {
		c.DetHighThresh = BaseConfig.DetHighThresh
	}
	if c.DetLowThresh == 0 {
		c.DetLowThresh = BaseConfig.DetLowThresh
	}
	if c.MatchDetHighThresh == 0 {
		c.MatchDetHighThresh = BaseConfig.MatchDetHighThresh
	}
	if c.MatchDetLowThresh == 0 {
		c.MatchDetLowThresh = BaseConfig.MatchDetLowThresh
	}
	if c.UnconfMatchThresh == 0 {
		c.UnconfMatchThresh = BaseConfig.UnconfMatchThresh
	}
	if c.RemoveDuplicateThresh == 0 {
		c.RemoveDuplicateThresh = BaseConfig.RemoveDuplicateThresh
	}
	if c.MaxFramesLost == 0 {
		c.MaxFramesLost = BaseConfig.MaxFramesLost
	}
	if c.MaxFramesReidable == 0 {
		c.MaxFramesReidable = BaseConfig.MaxFramesReidable
	}
	if c.IoUEmbThresh == 0 {
		c.IoUEmbThresh = BaseConfig.IoUEmbThresh
	}
	if c.EmbIoUThresh == 0 {
		c.EmbIoUThresh = BaseConfig.EmbIoUThresh
	}
	if c.EmbThresh == 0 {
		c.EmbThresh = BaseConfig.EmbThresh
	}
	if c.VRThresh == 0 {
		c.VRThresh = BaseConfig.VRThresh
	}
	if c.EmbReIDThresh == 0 {
		c.EmbReIDThresh = BaseConfig.EmbReIDThresh
	}
	if c.IoUReIDAlpha == 0 {
		c.IoUReIDAlpha = BaseConfig.IoUReIDAlpha
	}
	if c.MinPureCnt == 0 {
		c.MinPureCnt = BaseConfig.MinPureCnt
	}
	if c.EmbEMAAlpha == 0 {
		c.EmbEMAAlpha = BaseConfig.EmbEMAAlpha
	}
	if c.KFStdWeightProcessPos == 0 {
		c.KFStdWeightProcessPos = BaseConfig.KFStdWeightProcessPos
	}
	if c.KFStdWeightProcessVel == 0 {
		c.KFStdWeightProcessVel = BaseConfig.KFStdWeightProcessVel
	}
	if c.KFStdWeightMeasurement == 0 {
		c.KFStdWeightMeasurement = BaseConfig.KFStdWeightMeasurement
	}
	if c.KFInitPosCovFactor == 0 {
		c.KFInitPosCovFactor = BaseConfig.KFInitPosCovFactor
	}
	if c.KFInitVelCovFactor == 0 {
		c.KFInitVelCovFactor = BaseConfig.KFInitVelCovFactor
	}
}

const initPoolSize = 64

// Tracker is the public tracker instance. The type parameter T is the
// caller's concrete detection type. The optional sub-interfaces
// (DetectionWithEmbedding, DetectionWithClass, DetectionWithIndex) are
// detected per-detection via type assertion, so a single Tracker can mix
// detection variants without a separate constructor.
type Tracker[T Detection] struct {
	Config

	kf        *kalman.Filter
	ema       embedding.Handler
	lapSolver lap.Solver
	// iouBuf is reused across all four association cost-matrix builds
	// within one Update: first/second/unconfirmed association, plus
	// removeDuplicates. The calls are strictly sequential so sharing the
	// buffer is safe.
	iouBuf [][]float64

	// Pools keyed by track id. Removed tracks are never retained: Update
	// returns each one exactly once on the frame it transitions out.
	active   map[uint]*track[T]
	lost     map[uint]*track[T]
	reidable map[uint]*track[T]

	nextID  uint // monotonically increasing track id; starts at 1
	frameID uint // 1-based frame counter; 0 before the first Update

	imageW, imageH int // 0 disables the area component of the pure filter
}

// New creates a Tracker with the given configuration. Missing numeric
// fields are filled from BaseConfig.
func New[T Detection](config Config) *Tracker[T] {
	config.fillDefaults()
	return &Tracker[T]{
		Config: config,
		kf: kalman.New(kalman.Config{
			StdProcessPos:    config.KFStdWeightProcessPos,
			StdProcessVel:    config.KFStdWeightProcessVel,
			StdMeasurement:   config.KFStdWeightMeasurement,
			InitPosCovFactor: config.KFInitPosCovFactor,
			InitVelCovFactor: config.KFInitVelCovFactor,
		}),
		ema: embedding.Handler{
			Mode:     config.EmbMode,
			EMAAlpha: config.EmbEMAAlpha,
		},
		active:   make(map[uint]*track[T], initPoolSize),
		lost:     make(map[uint]*track[T], initPoolSize),
		reidable: make(map[uint]*track[T], initPoolSize),
		nextID:   1,
		frameID:  0,
	}
}

// SetImageSize enables the area component of the "pure detection" filter.
// When the image size is left at zero, only visibility ratio is used — a
// safe but looser gate.
func (t *Tracker[T]) SetImageSize(w, h int) {
	t.imageW, t.imageH = w, h
}

// FrameID returns the 1-based index of the most recently processed frame,
// or 0 if Update has not yet been called.
func (t *Tracker[T]) FrameID() uint { return t.frameID }

// Update runs one tracking step on the given detections. It returns the
// activated tracks for this frame (i.e. confirmed tracks that were updated
// this frame) and the tracks that transitioned to Removed. Each removed
// track is returned exactly once.
func (t *Tracker[T]) Update(detsXYXY []T) ([]Track[T], []Track[T]) {
	t.frameID++
	return t.runPipeline(detsXYXY)
}
