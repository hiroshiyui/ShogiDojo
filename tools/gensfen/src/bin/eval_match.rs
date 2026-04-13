//! eval_match: Measure model strength by playing games between two evaluators.
//!
//! Plays a configurable number of games between two models (or model vs. PST
//! baseline). Each pair of games swaps colors for fairness. Reports win/loss/draw
//! counts, win rates, and estimated Elo difference.
//!
//! Usage:
//!   eval_match --model-a model_v1.nnue --games 100 --depth 4
//!   eval_match --model-a model_v1.nnue --model-b model_v2.nnue --games 200

use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use clap::Parser;
use gensfen::eval::{Evaluator, PstEvaluator};
use gensfen::match_game::play_match_game;
use gensfen::nnue::NnueModel;
use gensfen::types::GameOutcome;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use shogi::bitboard::Factory as BBFactory;

/// Evaluate model strength by playing games between two evaluators.
#[derive(Parser)]
#[command(name = "eval_match", version, about)]
struct Args {
    /// Path to the first NNUE model (.nnue file).
    /// This is the model being evaluated ("Model A").
    #[arg(long)]
    model_a: String,

    /// Path to the second NNUE model (.nnue file).
    /// If omitted, Model A plays against the built-in PST evaluator.
    #[arg(long)]
    model_b: Option<String>,

    /// Total number of games to play.
    /// Games are played in pairs (A=Black + A=White) for fairness,
    /// so this is rounded up to the next even number.
    #[arg(long, default_value_t = 100)]
    games: usize,

    /// Search depth for both players.
    #[arg(long, default_value_t = 4)]
    depth: i32,

    /// Number of random opening moves per game.
    #[arg(long, default_value_t = 8)]
    random_moves: usize,

    /// Number of parallel threads.
    #[arg(long, default_value_t = 4)]
    threads: usize,
}

/// Accumulated match results (thread-safe via atomics).
struct MatchResults {
    a_wins: AtomicUsize,
    b_wins: AtomicUsize,
    draws: AtomicUsize,
}

impl MatchResults {
    fn new() -> Self {
        Self {
            a_wins: AtomicUsize::new(0),
            b_wins: AtomicUsize::new(0),
            draws: AtomicUsize::new(0),
        }
    }
}

/// Estimate Elo difference from win rate.
///
/// Uses the standard formula: Elo_diff = -400 * log10(1/win_rate - 1)
/// where win_rate includes draws counted as 0.5.
fn elo_difference(wins: usize, losses: usize, draws: usize) -> f64 {
    let total = (wins + losses + draws) as f64;
    if total == 0.0 {
        return 0.0;
    }
    let score = (wins as f64 + draws as f64 * 0.5) / total;
    if score <= 0.0 || score >= 1.0 {
        // Avoid log(0) or log(negative). Return a large value.
        return if score >= 1.0 { 800.0 } else { -800.0 };
    }
    -400.0 * (1.0 / score - 1.0).log10()
}

fn main() {
    BBFactory::init();

    let args = Args::parse();

    // Configure thread pool.
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .expect("failed to build rayon thread pool");

    // Load evaluators.
    let eval_a: Box<dyn Evaluator> = {
        eprintln!("Model A: {}", args.model_a);
        let model = NnueModel::load(Path::new(&args.model_a))
            .unwrap_or_else(|e| panic!("Failed to load model A: {e}"));
        eprintln!("  FT: {} → {}", model.ft_input_dim(), model.ft_out_dim());
        Box::new(model)
    };

    let eval_b: Box<dyn Evaluator> = match &args.model_b {
        Some(path) => {
            eprintln!("Model B: {}", path);
            let model = NnueModel::load(Path::new(path))
                .unwrap_or_else(|e| panic!("Failed to load model B: {e}"));
            eprintln!("  FT: {} → {}", model.ft_input_dim(), model.ft_out_dim());
            Box::new(model)
        }
        None => {
            eprintln!("Model B: built-in PST evaluator");
            Box::new(PstEvaluator)
        }
    };

    // Round up to even number for fair color swapping.
    let num_pairs = args.games.div_ceil(2);
    let total_games = num_pairs * 2;

    eprintln!();
    eprintln!(
        "Playing {} games ({} pairs, depth={}, random_moves={}, threads={})",
        total_games, num_pairs, args.depth, args.random_moves, args.threads,
    );

    let results = MatchResults::new();
    let games_played = AtomicUsize::new(0);
    let progress_lock = Mutex::new(());

    // Play game pairs in parallel. Each pair: A=Black then A=White.
    (0..num_pairs).into_par_iter().for_each(|pair_id| {
        let mut rng = SmallRng::seed_from_u64(pair_id as u64 ^ 0xBEEF_CAFE_1234);

        // Game 1: A plays Black, B plays White.
        let outcome1 = play_match_game(&*eval_a, &*eval_b, args.depth, args.random_moves, &mut rng);
        match outcome1 {
            GameOutcome::BlackWin => results.a_wins.fetch_add(1, Ordering::Relaxed),
            GameOutcome::WhiteWin => results.b_wins.fetch_add(1, Ordering::Relaxed),
            GameOutcome::Draw => results.draws.fetch_add(1, Ordering::Relaxed),
        };

        // Game 2: A plays White, B plays Black.
        let outcome2 = play_match_game(&*eval_b, &*eval_a, args.depth, args.random_moves, &mut rng);
        match outcome2 {
            GameOutcome::BlackWin => results.b_wins.fetch_add(1, Ordering::Relaxed),
            GameOutcome::WhiteWin => results.a_wins.fetch_add(1, Ordering::Relaxed),
            GameOutcome::Draw => results.draws.fetch_add(1, Ordering::Relaxed),
        };

        let played = games_played.fetch_add(2, Ordering::Relaxed) + 2;

        // Progress report every 10 games.
        if played % 10 < 2 {
            let _lock = progress_lock.lock().unwrap();
            eprintln!("  Progress: {}/{} games", played, total_games,);
        }
    });

    // Final results.
    let a_wins = results.a_wins.load(Ordering::Relaxed);
    let b_wins = results.b_wins.load(Ordering::Relaxed);
    let draws = results.draws.load(Ordering::Relaxed);
    let total = a_wins + b_wins + draws;
    let elo = elo_difference(a_wins, b_wins, draws);

    eprintln!();
    println!("=== Match Results ===");
    println!("Games played: {total}");
    println!();
    println!(
        "Model A wins: {a_wins:>5}  ({:.1}%)",
        a_wins as f64 / total as f64 * 100.0
    );
    println!(
        "Model B wins: {b_wins:>5}  ({:.1}%)",
        b_wins as f64 / total as f64 * 100.0
    );
    println!(
        "Draws:        {draws:>5}  ({:.1}%)",
        draws as f64 / total as f64 * 100.0
    );
    println!();
    println!("Elo difference (A vs B): {elo:+.0}");
}
