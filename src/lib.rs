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
