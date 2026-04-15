//! gensfen: Generate NNUE training data via Shogi self-play.
//!
//! This tool plays Shogi games against itself using alpha-beta search with
//! a material-only evaluation function, and records positions as training
//! data in text SFEN format. The output is directly consumable by
//! ShogiDojo's Python training pipeline.
//!
//! Usage:
//!   gensfen --depth 4 --threads 4 --positions 100000 --output data/train.sfen

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use clap::Parser;
use gensfen::eval::{Evaluator, PstEvaluator};
use gensfen::game;
use gensfen::nnue::{NnueEvaluator, NnueModel};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use shogi::bitboard::Factory as BBFactory;

/// Generate NNUE training data via Shogi self-play.
#[derive(Parser)]
#[command(name = "gensfen", version, about)]
struct Args {
    /// Search depth for the alpha-beta engine.
    /// Higher depth = stronger play and better quality data, but slower.
    /// Recommended: 3-4 for Phase 1 (material eval), 5-6 with NNUE eval.
    #[arg(long, default_value_t = 4)]
    depth: i32,

    /// Number of parallel threads for game generation.
    /// Each thread plays independent games with its own search state.
    #[arg(long, default_value_t = 4)]
    threads: usize,

    /// Total number of positions to generate.
    /// Each self-play game produces roughly 100-200 positions, so this
    /// determines how many games are played.
    #[arg(long)]
    positions: usize,

    /// Output file path for the training data (text SFEN format).
    #[arg(long, default_value = "output.sfen")]
    output: String,

    /// Number of random moves at the start of each game.
    /// Creates opening diversity so the training data covers a wide
    /// range of positions, not just the standard opening.
    #[arg(long, default_value_t = 8)]
    random_moves: usize,

    /// Path to a trained NNUE model file (.nnue) exported by export_nnue.py.
    /// If provided, uses the NNUE for evaluation instead of the built-in
    /// material + PST evaluator. This enables the iterative training loop:
    /// generate data → train NNUE → export → regenerate with stronger eval.
    #[arg(long)]
    model: Option<String>,
}

fn main() {
    // Must be called once before using any shogi-rs board functionality.
    // Initializes internal lookup tables for bitboard operations.
    BBFactory::init();

    let args = Args::parse();

    // Configure the rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .expect("failed to build rayon thread pool");

    // Load a shared NNUE model once (if requested). The `Arc` is handed to
    // each worker thread, which wraps it in its own `NnueEvaluator` with a
    // thread-local accumulator.
    let shared_model: Option<Arc<NnueModel>> = match &args.model {
        Some(model_path) => {
            eprintln!("Loading NNUE model: {}", model_path);
            let model = NnueModel::load(Path::new(model_path))
                .unwrap_or_else(|e| panic!("Failed to load model: {}", e));
            eprintln!("  FT: {} → {}", model.ft_input_dim(), model.ft_out_dim());
            Some(Arc::new(model))
        }
        None => {
            eprintln!("Using built-in PST evaluator (no --model specified)");
            None
        }
    };

    // Factory closure run inside each worker to build its own evaluator.
    let make_evaluator = || -> Box<dyn Evaluator> {
        match &shared_model {
            Some(m) => Box::new(NnueEvaluator::new(Arc::clone(m))),
            None => Box::new(PstEvaluator),
        }
    };

    // Open output file
    let file = File::create(&args.output).expect("failed to create output file");
    let writer = Mutex::new(BufWriter::new(file));

    // Shared counter: how many positions we've generated so far.
    // All threads atomically increment this until we reach the target.
    let total_positions = AtomicUsize::new(0);

    eprintln!(
        "Generating {} positions (depth={}, threads={}, random_moves={})",
        args.positions, args.depth, args.threads, args.random_moves
    );
    eprintln!("Output: {}", args.output);

    // Run game generation in parallel.
    // Each thread plays games until the global position counter reaches the target.
    (0..args.threads).into_par_iter().for_each(|thread_id| {
        // Each thread gets its own RNG seeded differently.
        let mut rng = SmallRng::seed_from_u64(thread_id as u64 ^ 0xDEAD_BEEF_CAFE);

        // Per-thread evaluator. For NNUE this owns the thread-local
        // accumulator; for PST it is trivial.
        let mut evaluator = make_evaluator();

        loop {
            // Check if we've already generated enough positions
            let current = total_positions.load(Ordering::Relaxed);
            if current >= args.positions {
                break;
            }

            // Play one complete game
            let samples = game::play_game(args.depth, args.random_moves, &mut rng, &mut *evaluator);

            if samples.is_empty() {
                continue; // Game produced no useful data, try again
            }

            // Write samples to the output file
            let count = samples.len();
            {
                let mut w = writer.lock().unwrap();
                for sample in &samples {
                    writeln!(w, "{}", sample.to_line()).unwrap();
                }
            }

            // Update the global counter
            let new_total = total_positions.fetch_add(count, Ordering::Relaxed) + count;

            // Progress report every ~10,000 positions
            if new_total / 10_000 > (new_total - count) / 10_000 {
                eprintln!(
                    "  Progress: {}/{} positions ({:.1}%)",
                    new_total,
                    args.positions,
                    new_total as f64 / args.positions as f64 * 100.0
                );
            }
        }
    });

    let final_count = total_positions.load(Ordering::Relaxed);
    eprintln!("Done. Generated {} positions.", final_count);
}
