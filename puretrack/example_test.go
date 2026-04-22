package puretrack_test

import (
	"fmt"

	"github.com/ugparu/gopuretrack/puretrack"
)

// exampleDetection is the smallest type that satisfies puretrack.Detection:
// a bounding box in (x1, y1, x2, y2) and a confidence score.
type exampleDetection struct {
	xyxy  [4]float64
	score float64
}

func (d exampleDetection) GetXYXY() [4]float64 { return d.xyxy }
func (d exampleDetection) GetScore() float64   { return d.score }

// ExampleTracker shows the two-call shape of the public API: construct a
// tracker with the default configuration, then feed detections one frame
// at a time. Update returns the confirmed tracks matched on this frame.
func ExampleTracker() {
	tracker := puretrack.New[exampleDetection](puretrack.BaseConfig)

	active, _ := tracker.Update([]exampleDetection{
		{xyxy: [4]float64{100, 100, 150, 200}, score: 0.9},
	})
	for _, t := range active {
		fmt.Printf("id=%d bbox=%v\n", t.GetID(), t.GetXYXY())
	}
	// Output:
	// id=1 bbox=[100 100 150 200]
}
