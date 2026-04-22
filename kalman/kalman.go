// Package kalman implements a constant-velocity Kalman filter on bounding
// boxes in (x, y, w, h) coordinates. The state vector is
// [x, y, w, h, vx, vy, vw, vh] and the measurement vector is [x, y, w, h].
//
// The filter supports NSA (Noise Scale Adaptation, Du et al., GIAOTracker):
// per-measurement noise R is scaled by (1 - conf)² so that low-confidence
// detections contribute less to the posterior.
package kalman

import (
	"unsafe"

	"gonum.org/v1/gonum/mat"
)

// ndim is the number of spatial dimensions. The state vector is 2·ndim
// (position + velocity) and the measurement vector is ndim.
const ndim = 4

// Config holds the tunable noise weights for the filter.
type Config struct {
	// Process-noise standard deviations, expressed as a fraction of the
	// current bbox width/height. Applied each predict step.
	StdProcessPos  float64
	StdProcessVel  float64
	StdMeasurement float64

	// Multipliers applied to StdProcessPos / StdProcessVel when a track is
	// born. Larger values make the filter less confident in the initial
	// state, which widens the gate for the first association.
	InitPosCovFactor float64
	InitVelCovFactor float64
}

// Filter is a batched (x, y, w, h) Kalman filter. A Filter is stateless
// with respect to tracks — all per-track state is passed in by the caller —
// but not safe for concurrent use because the scratch matrices are shared.
type Filter struct {
	cfg Config

	motionMat, motionMatT *mat.Dense // 8×8 transition F and its transpose
	updateMat, updateMatT *mat.Dense // 4×8 measurement H and its transpose
}

// New builds a Filter from cfg. The transition matrix encodes a unit time
// step (dt = 1), so callers that need a different frame cadence should
// scale their process-noise weights accordingly.
func New(cfg Config) *Filter {
	motion := mat.NewDense(2*ndim, 2*ndim, nil)
	for i := range 2 * ndim {
		motion.Set(i, i, 1.0)
	}
	for i := range ndim {
		motion.Set(i, ndim+i, 1.0)
	}
	update := mat.NewDense(ndim, 2*ndim, nil)
	for i := range ndim {
		update.Set(i, i, 1.0)
	}
	return &Filter{
		cfg:        cfg,
		motionMat:  motion,
		motionMatT: denseT(motion),
		updateMat:  update,
		updateMatT: denseT(update),
	}
}

func denseT(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	out := mat.NewDense(c, r, nil)
	out.Copy(m.T())
	return out
}

// MultiInitiate writes birth means and covariances into the caller-supplied
// buffers for a batch of (x, y, w, h) measurements. The initial mean sets
// position to the measurement and velocity to zero. The covariance is
// diagonal with entries (init_factor · std · dim)² where dim cycles over
// [w, h, w, h] for both the position and velocity blocks.
func (f *Filter) MultiInitiate(measurements [][4]float64, means [][8]float64, covs [][64]float64) {
	posFactor := f.cfg.InitPosCovFactor * f.cfg.StdProcessPos
	velFactor := f.cfg.InitVelCovFactor * f.cfg.StdProcessVel
	for i, m := range measurements {
		copy(means[i][:4], m[:])
		means[i][4], means[i][5], means[i][6], means[i][7] = 0, 0, 0, 0

		for j := range covs[i] {
			covs[i][j] = 0
		}
		w, h := m[2], m[3]
		posStds := [4]float64{posFactor * w, posFactor * h, posFactor * w, posFactor * h}
		velStds := [4]float64{velFactor * w, velFactor * h, velFactor * w, velFactor * h}
		for j := range 4 {
			covs[i][j*8+j] = posStds[j] * posStds[j]
			covs[i][(j+4)*8+(j+4)] = velStds[j] * velStds[j]
		}
	}
}

// MultiPredict advances means and covariances in place:
//
//	x ← F · x,   P ← F · P · Fᵀ + Q
//
// Q is diagonal with entries
//
//	(std_pos · [w, h, w, h, std_vel·w, std_vel·h, …])²
//
// computed from the *pre-predict* width and height. Sampling Q before the
// state update preserves the association between a track's current size
// and its next-step uncertainty.
func (f *Filter) MultiPredict(means [][8]float64, covs [][64]float64) {
	n := len(means)
	if n == 0 {
		return
	}

	qDiags := make([][8]float64, n)
	pp := f.cfg.StdProcessPos
	pv := f.cfg.StdProcessVel
	for i := range n {
		w, h := means[i][2], means[i][3]
		qDiags[i] = [8]float64{
			sq(pp * w), sq(pp * h), sq(pp * w), sq(pp * h),
			sq(pv * w), sq(pv * h), sq(pv * w), sq(pv * h),
		}
	}

	// mean ← mean · Fᵀ. gonum's Mul handles the read/write aliasing safely.
	meanMat := mat.NewDense(n, 8, unsafe.Slice(&means[0][0], 8*n))
	meanMat.Mul(meanMat, f.motionMatT)

	left := mat.NewDense(8, 8, nil)
	right := mat.NewDense(8, 8, nil)
	for i := range n {
		qDiag := qDiags[i]

		covMat := mat.NewDense(8, 8, covs[i][:])
		left.Mul(f.motionMat, covMat)
		right.Mul(left, f.motionMatT)
		covMat.Copy(right)
		for j := range 8 {
			covs[i][j*8+j] += qDiag[j]
		}
	}
}

// MultiProject returns the projected means (4-vector) and covariances
// (4×4 flattened row-major) in measurement space. confs carries per-track
// detection confidence and enables NSA; a nil slice is equivalent to
// conf=0 for every track (no NSA scaling — R is used as-is).
func (f *Filter) MultiProject(means [][8]float64, covs [][64]float64, confs []float64) ([][4]float64, [][16]float64) {
	n := len(means)
	projMeans := make([][4]float64, n)
	projCovs := make([][16]float64, n)
	if n == 0 {
		return projMeans, projCovs
	}

	// H · x is just the position block of x.
	for i := range means {
		copy(projMeans[i][:], means[i][:4])
	}

	tmp := mat.NewDense(ndim, 2*ndim, nil)
	for i := range n {
		covMat := mat.NewDense(8, 8, covs[i][:])
		projCovMat := mat.NewDense(ndim, ndim, unsafe.Slice(&projCovs[i][0], 16))
		tmp.Mul(f.updateMat, covMat)
		projCovMat.Mul(tmp, f.updateMatT)

		conf := 0.0
		if confs != nil {
			conf = confs[i]
		}
		scale := (1 - conf) * (1 - conf)
		w, h := means[i][2], means[i][3]
		sm := f.cfg.StdMeasurement
		rDiag := [4]float64{sq(sm * w), sq(sm * h), sq(sm * w), sq(sm * h)}
		for j := range 4 {
			projCovs[i][j*4+j] += scale * rDiag[j]
		}
	}
	return projMeans, projCovs
}

// MultiUpdate applies the Kalman correction in place. means, covs and
// measurements must all have the same length. confs may be nil to disable
// NSA scaling of the measurement noise.
func (f *Filter) MultiUpdate(means [][8]float64, covs [][64]float64, measurements [][4]float64, confs []float64) {
	n := len(means)
	if n == 0 {
		return
	}

	projMeans, projCovs := f.MultiProject(means, covs, confs)

	covHT := mat.NewDense(8, ndim, nil)
	kalmanGain := mat.NewDense(8, ndim, nil)
	kalmanGainT := mat.NewDense(ndim, 8, nil)
	tmp84 := mat.NewDense(8, ndim, nil)
	tmp88 := mat.NewDense(8, 8, nil)

	for i := range n {
		covMat := mat.NewDense(8, 8, covs[i][:])

		// Symmetrize the projected covariance to absorb any accumulated
		// rounding asymmetry before the Cholesky factorization.
		projCovMat := mat.NewSymDense(ndim, nil)
		for r := range ndim {
			for c := r; c < ndim; c++ {
				v := 0.5 * (projCovs[i][r*ndim+c] + projCovs[i][c*ndim+r])
				projCovMat.SetSym(r, c, v)
			}
		}

		covHT.Mul(covMat, f.updateMatT)

		// Solve S · Kᵀ = (P · Hᵀ)ᵀ for the Kalman gain K.
		var ch mat.Cholesky
		if !ch.Factorize(projCovMat) {
			// Near-singular projection — add a tiny ridge and retry.
			for j := range ndim {
				projCovMat.SetSym(j, j, projCovMat.At(j, j)+1e-9)
			}
			if !ch.Factorize(projCovMat) {
				// Still singular after the ridge — skip the update for this
				// track rather than feed garbage through the state. The
				// filter will have another chance on the next observation.
				continue
			}
		}
		if err := ch.SolveTo(kalmanGainT, covHT.T()); err != nil {
			continue
		}
		kalmanGain.Copy(kalmanGainT.T())

		var innov [4]float64
		for j := range ndim {
			innov[j] = measurements[i][j] - projMeans[i][j]
		}

		var addMean [8]float64
		for r := range 8 {
			s := 0.0
			for c := range ndim {
				s += kalmanGain.At(r, c) * innov[c]
			}
			addMean[r] = s
		}
		for r := range 8 {
			means[i][r] += addMean[r]
		}

		// P ← P - K · S · Kᵀ.
		projCovDense := mat.NewDense(ndim, ndim, nil)
		for r := range ndim {
			for c := range ndim {
				projCovDense.Set(r, c, projCovMat.At(r, c))
			}
		}
		tmp84.Mul(kalmanGain, projCovDense)
		tmp88.Mul(tmp84, kalmanGainT)
		covMat.Sub(covMat, tmp88)
	}
}

func sq(x float64) float64 { return x * x }
