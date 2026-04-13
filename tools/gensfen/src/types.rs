//! Shared data types used across the gensfen tool.

use shogi::Color;

/// The outcome of a completed self-play game, from an absolute perspective.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GameOutcome {
    BlackWin,
    WhiteWin,
    Draw,
}

/// A position recorded during a self-play game.
///
/// At recording time, we know the SFEN and the search score, but we don't
/// yet know the game result (the game hasn't finished). The result is
/// "backfilled" after the game ends — see `resolve()`.
pub struct RecordedPosition {
    /// SFEN string of the position (e.g., "lnsgkgsnl/... b - 42").
    pub sfen: String,
    /// Search evaluation in centipawns, from the side-to-move's perspective.
    /// Positive = good for side-to-move.
    pub score: i32,
    /// Which player was to move in this position.
    pub side_to_move: Color,
}

/// A fully resolved training sample, ready to be written to the output file.
///
/// This is what `dataset.py` in ShogiDojo will parse. Each sample becomes
/// one line in the output file:
///   `sfen <board> <side> <hand> <move#>\t<score>\t<result>`
pub struct TrainingSample {
    pub sfen: String,
    pub score: i32,
    /// Game result from the side-to-move's perspective:
    ///   1.0 = side-to-move won, 0.5 = draw, 0.0 = side-to-move lost.
    pub result: f32,
}

impl RecordedPosition {
    /// Convert this recorded position into a training sample by filling in
    /// the game result.
    ///
    /// The result is relative to `self.side_to_move`:
    ///   - If side_to_move won → 1.0
    ///   - If side_to_move lost → 0.0
    ///   - Draw → 0.5
    pub fn resolve(self, outcome: GameOutcome) -> TrainingSample {
        let result = match (outcome, self.side_to_move) {
            (GameOutcome::BlackWin, Color::Black) => 1.0,
            (GameOutcome::BlackWin, Color::White) => 0.0,
            (GameOutcome::WhiteWin, Color::White) => 1.0,
            (GameOutcome::WhiteWin, Color::Black) => 0.0,
            (GameOutcome::Draw, _) => 0.5,
        };
        TrainingSample {
            sfen: self.sfen,
            score: self.score,
            result,
        }
    }
}

impl TrainingSample {
    /// Format this sample as a line for the training data file.
    ///
    /// Output format (tab-separated score and result):
    ///   `sfen lnsgkgsnl/... b - 1\t100\t1.0`
    pub fn to_line(&self) -> String {
        format!("sfen {}\t{}\t{:.1}", self.sfen, self.score, self.result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // RecordedPosition::resolve — all outcome × side-to-move combinations
    // -----------------------------------------------------------------------

    #[test]
    fn resolve_black_wins_from_black_perspective() {
        let rec = RecordedPosition {
            sfen: "test".into(),
            score: 100,
            side_to_move: Color::Black,
        };
        let sample = rec.resolve(GameOutcome::BlackWin);
        assert_eq!(sample.result, 1.0);
        assert_eq!(sample.score, 100);
    }

    #[test]
    fn resolve_black_wins_from_white_perspective() {
        let rec = RecordedPosition {
            sfen: "test".into(),
            score: -50,
            side_to_move: Color::White,
        };
        let sample = rec.resolve(GameOutcome::BlackWin);
        assert_eq!(sample.result, 0.0);
    }

    #[test]
    fn resolve_white_wins_from_white_perspective() {
        let rec = RecordedPosition {
            sfen: "test".into(),
            score: 200,
            side_to_move: Color::White,
        };
        let sample = rec.resolve(GameOutcome::WhiteWin);
        assert_eq!(sample.result, 1.0);
    }

    #[test]
    fn resolve_white_wins_from_black_perspective() {
        let rec = RecordedPosition {
            sfen: "test".into(),
            score: -200,
            side_to_move: Color::Black,
        };
        let sample = rec.resolve(GameOutcome::WhiteWin);
        assert_eq!(sample.result, 0.0);
    }

    #[test]
    fn resolve_draw_from_either_perspective() {
        for color in [Color::Black, Color::White] {
            let rec = RecordedPosition {
                sfen: "test".into(),
                score: 0,
                side_to_move: color,
            };
            let sample = rec.resolve(GameOutcome::Draw);
            assert_eq!(sample.result, 0.5);
        }
    }

    // -----------------------------------------------------------------------
    // TrainingSample::to_line — output format
    // -----------------------------------------------------------------------

    #[test]
    fn to_line_format() {
        let sample = TrainingSample {
            sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1".into(),
            score: 100,
            result: 1.0,
        };
        let line = sample.to_line();
        assert!(line.starts_with("sfen "));
        // Tab-separated score and result
        let parts: Vec<&str> = line.split('\t').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[1], "100");
        assert_eq!(parts[2], "1.0");
    }

    #[test]
    fn to_line_negative_score() {
        let sample = TrainingSample {
            sfen: "test".into(),
            score: -300,
            result: 0.0,
        };
        let line = sample.to_line();
        let parts: Vec<&str> = line.split('\t').collect();
        assert_eq!(parts[1], "-300");
        assert_eq!(parts[2], "0.0");
    }

    #[test]
    fn to_line_draw_result() {
        let sample = TrainingSample {
            sfen: "test".into(),
            score: 0,
            result: 0.5,
        };
        let line = sample.to_line();
        let parts: Vec<&str> = line.split('\t').collect::<Vec<_>>();
        assert_eq!(parts[2], "0.5");
    }

    #[test]
    fn resolve_preserves_sfen() {
        let sfen = "4k4/9/9/9/9/9/9/9/4K4 b - 1".to_string();
        let rec = RecordedPosition {
            sfen: sfen.clone(),
            score: 42,
            side_to_move: Color::Black,
        };
        let sample = rec.resolve(GameOutcome::Draw);
        assert_eq!(sample.sfen, sfen);
        assert_eq!(sample.score, 42);
    }
}
