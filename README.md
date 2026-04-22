# GoPureTrack

[![CI](https://github.com/ugparu/gopuretrack/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/ugparu/gopuretrack/actions/workflows/ci.yml)
[![Lint](https://github.com/ugparu/gopuretrack/actions/workflows/lint.yml/badge.svg?branch=master)](https://github.com/ugparu/gopuretrack/actions/workflows/lint.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/ugparu/gopuretrack.svg)](https://pkg.go.dev/github.com/ugparu/gopuretrack)
[![Go Report Card](https://goreportcard.com/badge/github.com/ugparu/gopuretrack)](https://goreportcard.com/report/github.com/ugparu/gopuretrack)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

High-performance multi-object tracker in pure Go: constant-velocity Kalman
filter on bounding boxes with optional appearance-based re-identification.

![MOT17-04 demo](examples/MOT17-04.gif)

## Why GoPureTrack

- **Pure Go, no cgo.** Cross-compiles anywhere Go does; no C toolchain
  needed at build time.
- **Generic over your detection type.** `Tracker[T]` keeps your detection
  struct, so there is no per-frame conversion boilerplate.
- **Deterministic and verified.** Numerical parity against a reference
  implementation is tested per-frame on seven MOT17 sequences; bbox error
  stays within float32 rounding (~7e-5 ℓ∞).
- **Handles occlusion and id switches.** Two-stage association cascade
  (high- then low-confidence), unconfirmed-track probation, long-range
  appearance re-identification.

## Quick Start

```bash
go get github.com/ugparu/gopuretrack
```

```go
package main

import (
	"fmt"

	"github.com/ugparu/gopuretrack/puretrack"
)

// Detection is your own type. The only required methods are GetXYXY and
// GetScore; the tracker discovers optional capabilities (class, detection
// id, appearance embedding) via type assertion.
type Detection struct {
	XYXY  [4]float64
	Score float64
}

func (d Detection) GetXYXY() [4]float64 { return d.XYXY }
func (d Detection) GetScore() float64   { return d.Score }

func main() {
	tracker := puretrack.New[Detection](puretrack.BaseConfig)

	// One frame of detections (x1, y1, x2, y2, score).
	dets := []Detection{
		{XYXY: [4]float64{100, 100, 150, 200}, Score: 0.92},
		{XYXY: [4]float64{400, 220, 460, 340}, Score: 0.81},
	}

	active, removed := tracker.Update(dets)
	for _, t := range active {
		fmt.Printf("id=%d bbox=%v\n", t.GetID(), t.GetXYXY())
	}
	_ = removed // tracks that transitioned to Removed this frame
}
```

Every call to `Update` advances one frame. The first return value is the
set of confirmed tracks that were matched to a detection on this frame;
the second is the set of tracks that transitioned to Removed and will
not be seen again. Tracks that merely missed the current frame stay
internally in the Lost pool and do not appear in either slice — they can
resurface in a later `Update` if a matching detection arrives.

## Documentation

Full API reference on [pkg.go.dev](https://pkg.go.dev/github.com/ugparu/gopuretrack).

Key types:

- [`puretrack.Tracker[T]`](https://pkg.go.dev/github.com/ugparu/gopuretrack/puretrack#Tracker)
  — the main entry point.
- [`puretrack.Config`](https://pkg.go.dev/github.com/ugparu/gopuretrack/puretrack#Config)
  and [`puretrack.BaseConfig`](https://pkg.go.dev/github.com/ugparu/gopuretrack/puretrack#pkg-variables)
  — tunable knobs, with zero-value fields filled from defaults.
- [`puretrack.Track[T]`](https://pkg.go.dev/github.com/ugparu/gopuretrack/puretrack#Track) /
  [`TrackExt[T]`](https://pkg.go.dev/github.com/ugparu/gopuretrack/puretrack#TrackExt)
  — the tracked-object view returned by `Update`.
- [`puretrack.DetectionWithEmbedding`](https://pkg.go.dev/github.com/ugparu/gopuretrack/puretrack#DetectionWithEmbedding)
  — optional interface for re-identification mode.

## Features

- Constant-velocity Kalman filter on (x, y, w, h) with NSA
  measurement-noise scaling (per-track R scaled by `(1 - conf)²`).
- Two-stage association cascade: confirmed tracks × high-confidence
  detections first, then remaining Tracked × low-confidence detections.
- Unconfirmed-track probation: single-frame spurious detections are
  rejected before becoming a persistent id.
- Long-range re-identification of Reidable tracks via pure appearance
  embeddings (EMA-smoothed, confidence-weighted).
- LAPJV assignment solver, pure Go.
- Generic `Tracker[T]` over any detection struct.
- MOT17 numerical parity test suite: seven sequences without ReID,
  MOT17-09 with ReID (full 262-frame replay; other sequences omit the
  ReID fixture because each exceeds ~200 MB).

## How It Works

Each call to `Update` runs the pipeline below. The order is chosen so
that high-confidence, motion-consistent matches happen before any
fallback; new tracks are only spawned from detections nothing else
claimed.

```text
Update(dets)
  │
  ├─▶ prepare detections       split by confidence (high / low),
  │                            compute visibility ratios
  │
  ├─▶ prepare tracks           Kalman predict for every matchable track
  │
  ├─▶ re-identification  [opt] Reidable track × active track by pure
  │                            embedding distance → merge older id
  │
  ├─▶ first association        Tracked/Lost × high-conf dets
  │                            fused IoU + (optional) embedding cost
  │
  ├─▶ second association       remaining Tracked × low-conf dets
  │                            pure IoU cost — rescues occluders
  │
  ├─▶ unconfirmed handling     match or retire one-frame candidates
  │
  ├─▶ initialize new tracks    spawn ids from leftover high-conf dets
  │
  └─▶ update state             active / lost / reidable pools,
                               duplicate suppression, output assembly
```

Track lifecycle: `New → Tracked ↔ Lost → Reidable → Removed`.
A Lost track is revived by a matching detection; past `MaxFramesLost`
it becomes Reidable (if it accumulated enough unoccluded observations)
or is Removed outright.

## Configuration

The defaults in [`BaseConfig`](https://pkg.go.dev/github.com/ugparu/gopuretrack/puretrack#pkg-variables)
are tuned for 30 fps pedestrian tracking. Any zero-valued numeric field
is filled from `BaseConfig`, so you only set what you want to override.

| Field | Default | What it does |
|---|---|---|
| `TrackNewThresh` | 0.6 | Minimum score to spawn a new track |
| `DetHighThresh` | 0.6 | High/low detection-pool split |
| `DetLowThresh` | 0.1 | Floor for the low-confidence pool |
| `MatchDetHighThresh` | 0.9 | Cost ceiling in the first association |
| `MatchDetLowThresh` | 0.5 | Cost ceiling in the second association |
| `UnconfMatchThresh` | 0.7 | Cost ceiling for unconfirmed matches |
| `MaxFramesLost` | 36 | Frames a missed track stays Lost before aging out (≈1.2 s @ 30 fps) |
| `MaxFramesReidable` | 150 | Frames a Reidable track is kept for ReID (≈5 s @ 30 fps) |
| `RemoveDuplicateThresh` | 0.15 | 1 - IoU threshold for duplicate pruning |
| `WithReID` | false | Enable appearance-based matching |
| `WithEmbReactivation` | true | Enable Reidable pool and long-range ReID |
| `MinPureCnt` | 15 | Pure observations required before a track is ReID-eligible |

See [`Config`](https://pkg.go.dev/github.com/ugparu/gopuretrack/puretrack#Config)
for the full set (ReID thresholds, Kalman noise weights, EMA alpha).

## Re-identification Mode

To enable appearance-based matching, set `WithReID: true` and make your
detection type implement
[`DetectionWithEmbedding`](https://pkg.go.dev/github.com/ugparu/gopuretrack/puretrack#DetectionWithEmbedding).
The tracker feeds the embedding through an EMA and fuses cosine
distance with IoU cost in the first-association step.

```go
type Detection struct {
	XYXY  [4]float64
	Score float64
	Embed []float64 // feature vector from your ReID model
}

func (d Detection) GetXYXY() [4]float64    { return d.XYXY }
func (d Detection) GetScore() float64      { return d.Score }
func (d Detection) GetEmbedding() []float64 { return d.Embed }

cfg := puretrack.BaseConfig
cfg.WithReID = true
tracker := puretrack.New[Detection](cfg)

tracker.SetImageSize(1920, 1080) // enables the area component of the
                                 // "pure detection" filter used by ReID
```

## Benchmarks

Full-sequence replays on a fresh tracker. Machine: Intel i7-12700H,
Linux, Go 1.24.

| Sequence  | Mode    | Throughput      | Per-frame | Real-time @ 30 fps |
|-----------|---------|-----------------|-----------|--------------------|
| MOT17-02  | no ReID |  4 012 frames/s | 249 µs    | 134×               |
| MOT17-04  | no ReID |  2 520 frames/s | 397 µs    |  84×               |
| MOT17-05  | no ReID | 11 921 frames/s |  84 µs    | 397×               |
| MOT17-09  | no ReID | 16 039 frames/s |  62 µs    | 535×               |
| MOT17-10  | no ReID |  5 495 frames/s | 182 µs    | 183×               |
| MOT17-11  | no ReID | 10 366 frames/s |  97 µs    | 346×               |
| MOT17-13  | no ReID | 14 209 frames/s |  70 µs    | 474×               |
| MOT17-09  | ReID    |  2 375 frames/s | 421 µs    |  79×               |

Throughput scales inversely with detection density — more detections per
frame means more rows in the cost matrix and more embedding updates.
Enabling ReID on MOT17-09 adds cosine-distance computation, a second LAP
solve for long-range re-identification, and the EMA appearance update,
which together drop throughput by roughly 6.7× on the same sequence.

Reproduce:

```bash
go test -bench . -benchmem -run=^$ -benchtime=3s ./puretrack
```

## Development

```bash
go test ./...                          # unit + parity tests
go test -race ./...                    # with data-race detector
go test -bench . -benchmem ./puretrack # benchmarks
golangci-lint run ./...                # lint with the pinned v2 config
```

The parity suite under `puretrack/parity_test.go` replays every MOT17
sequence against a stored reference and fails if per-frame bbox ℓ∞
error exceeds 2×10⁻⁴ or any track id diverges.

## Releases

Version history and release notes are on the
[GitHub Releases page](https://github.com/ugparu/gopuretrack/releases).

## Contributing

Issues and pull requests are welcome. Please run `go test ./...` and
`golangci-lint run ./...` before opening a PR.

## License

MIT — see [LICENSE](./LICENSE).

## Acknowledgments

The two-stage high-/low-confidence association cascade is the core idea
from [ByteTrack](https://arxiv.org/abs/2110.06864). NSA measurement-noise
scaling follows the formulation popularized by
[StrongSORT](https://arxiv.org/abs/2202.13514). The LAPJV assignment
solver is a port of Jonker & Volgenant's 1987 algorithm. Linear algebra
is backed by [gonum](https://gonum.org/).
