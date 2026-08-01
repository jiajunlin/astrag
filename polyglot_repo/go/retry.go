package util

import (
	"math/rand"
	"time"
)

// Retry runs fn up to attempts times, sleeping with exponential backoff
// (plus jitter) between failures, and returns the last error.
func Retry(attempts int, base time.Duration, fn func() error) error {
	var err error
	for i := 0; i < attempts; i++ {
		if err = fn(); err == nil {
			return nil
		}
		sleep := base * time.Duration(1<<i)
		sleep += time.Duration(rand.Int63n(int64(base)))
		time.Sleep(sleep)
	}
	return err
}

// Backoff describes an exponential backoff schedule.
type Backoff struct {
	Base   time.Duration
	Factor float64
}

// Next returns the delay before attempt i (0-based).
func (b Backoff) Next(i int) time.Duration {
	d := float64(b.Base)
	for j := 0; j < i; j++ {
		d *= b.Factor
	}
	return time.Duration(d)
}
