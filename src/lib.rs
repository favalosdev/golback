//! # golback
//!
//! A high-performance Game of Life backend using the HashLife algorithm with quadtree compression.
//!
//! This crate provides an efficient implementation of Conway's Game of Life that can simulate
//! thousands of generations per second for many patterns through memoization and recursive
//! quadtree structures.
//!
//! ## Features
//!
//! - **High Performance**: Uses HashLife algorithm for exponential speedup
//! - **Memory Efficient**: Quadtree compression reduces memory usage
//! - **Flexible Input**: Load patterns from RLE (Run-Length Encoded) files or coordinate lists
//! - **Rule Support**: Configurable birth/survival rules (beyond standard B3/S23)
//! - **Pattern Editing**: Toggle individual cells, export current state
//!
//! ## Quick Start
//!
//! ```rust
//! use golback::universe::Universe;
//!
//! // Create and initialize a universe
//! let mut universe = Universe::new();
//! universe.init();
//!
//! // Load a pattern
//! universe.load("glider.rle".to_string()).unwrap();
//!
//! // Advance generations
//! universe.advance(100);
//!
//! // Check results
//! println!("Population: {}", universe.population());
//! ```

pub mod universe;

#[cfg(test)]
mod tests {
    use std::collections::LinkedList;
    use crate::universe::Universe;
    const TEST_DIM: u32 = 3;

    #[test]
    fn new_initializes_zero_population_and_epoch() {
        let universe = Universe::new();
        assert_eq!(universe.population(), 0);
        assert_eq!(universe.epochs(), 0);
    }

    #[test]
    fn add_delete_toggle_and_is_alive_work() {
        let mut u = Universe::new();
        u.init(TEST_DIM);

        u.add((0, 0));
        assert!(u.is_alive((0, 0)));
        assert_eq!(u.population(), 1);

        u.delete((0, 0));
        assert!(!u.is_alive((0, 0)));
        assert_eq!(u.population(), 0);

        u.toggle((1, 1));
        assert!(u.is_alive((1, 1)));
        assert_eq!(u.population(), 1);

        u.toggle((1, 1));
        assert!(!u.is_alive((1, 1)));
        assert_eq!(u.population(), 0);
    }

    #[test]
    fn from_coords_and_to_coords_roundtrip() {
        let mut u = Universe::new();
        u.init(TEST_DIM);

        let mut cells = LinkedList::new();
        cells.push_back((0, 0));
        cells.push_back((1, 0));
        cells.push_back((0, 1));
        cells.push_back((1, 1));

        u.from_coords(cells);
        assert_eq!(u.population(), 4);

        u.is_alive((0, 0));
        u.is_alive((1, 0));
        u.is_alive((0, 1));
        u.is_alive((1, 1));

        let coords = u.to_coords();
        assert_eq!(coords.len(), 4);
        assert!(coords.contains(&(0, 0)));
        assert!(coords.contains(&(1, 0)));
        assert!(coords.contains(&(0, 1)));
        assert!(coords.contains(&(1, 1)));
    }

    #[test]
    fn advance_performs_blinker_step() {
        let mut u = Universe::new();
        u.init(TEST_DIM);

        let mut cells = LinkedList::new();
        cells.push_back((-1, 0));
        cells.push_back((0, 0));
        cells.push_back((1, 0));

        u.from_coords(cells);
        assert_eq!(u.population(), 3);

        u.advance(1);

        assert_eq!(u.population(), 3);
        assert!(!u.is_alive((-1, 0)));
        assert!(u.is_alive((0, -1)));
        assert!(u.is_alive((0, 0)));
        assert!(u.is_alive((0, 1)));
    }
}
