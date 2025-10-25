#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <map>
#include <random>
#include <queue>

namespace py = pybind11;
// #include <iostream> //remove later,keep for debugging

#define bottom_score_row 10
#define top_score_row 2
#define rows 13
#define cols 12

using namespace std;

struct Piece // compact piece representation
{
    uint8_t owner;       // 0=none, 1=circle, 2=square
    uint8_t side;        // 0=stone, 1=river
    uint8_t orientation; // 0=horizontal, 1=vertical

    bool empty() const { return owner == 0; }
    bool is_stone() const { return side == 0; }
    bool is_river() const { return side == 1; }
    bool is_circle() const { return owner == 1; }
    bool is_square() const { return owner == 2; }
    bool is_horizontal() const { return orientation == 0; }
    bool is_vertical() const { return orientation == 1; }
};

using FastBoard = std::vector<std::vector<Piece>>;

struct Move // general move structure
{
    std::string action;
    std::vector<int> from;
    std::vector<int> to;
    std::vector<int> pushed_to;
    std::string orientation;
};

struct ScoredMove
{
    Move move;
    float score;
};

struct SavedPiece // for making and undoing moves
{
    std::vector<int> pos;
    Piece data;
};

struct EvalFeatures
{
    float stones_in_goal = 0;
    float immediate_threats = 0;
    float one_move_away = 0;
    float advancement_sum = 0;
    float center_control = 0;
    float river_support = 0;
    float mobility = 0;
    float blocking = 0;
};

struct EvalWeights
{
    float w_stones_goal = 100.0f;
    float w_threats = 5.0f;
    float w_one_away = 40.0f;
    float w_advancement = 5.0f;
    float w_center = -30.0f;
    float w_river = 15.0f;
    float w_mobility = 20.0f;
    float w_blocking = 30.0f;
};

struct MoveTargets
{
    vector<pair<int, int>> moves;
    vector<pair<pair<int, int>, pair<int, int>>> pushes; // ((tx,ty), (px,py))
};

void make_move(FastBoard &board, const Move &move, std::vector<SavedPiece> &saved_pieces);
void undo_move(FastBoard &board, const std::vector<SavedPiece> &saved_pieces);

FastBoard convert_to_fast_board(const vector<vector<map<string, string>>> &slow_board)
{
    FastBoard fast(slow_board.size(), vector<Piece>(slow_board[0].size()));
    for (size_t y = 0; y < slow_board.size(); y++)
    {
        for (size_t x = 0; x < slow_board[y].size(); x++)
        {
            if (slow_board[y][x].empty())
            {
                fast[y][x] = {0, 0, 0};
            }
            else
            {
                uint8_t owner = (slow_board[y][x].at("owner") == "circle") ? 1 : 2;
                uint8_t side = (slow_board[y][x].at("side") == "stone") ? 0 : 1;
                uint8_t orientation = (slow_board[y][x].at("orientation") == "horizontal") ? 0 : 1;
                fast[y][x] = {owner, side, orientation};
            }
        }
    }

    return fast;
}

std::string opponent(const std::string &player)
{
    return (player == "circle") ? "square" : "circle";
}

const float distance_values[rows][cols] = {{6.398, 10.549, 15.737, 21.242, 25.945, 28.674, 28.674, 25.945, 21.242, 15.737, 10.549, 6.398},
                                           {12.379, 20.409, 30.447, 41.099, 50.199, 55.478, 55.478, 50.199, 41.099, 30.447, 20.409, 12.379},
                                           {15.425, 25.432, 37.940, 51.213, 100, 100, 100, 100, 51.213, 37.940, 25.432, 15.425},
                                           {14.673, 24.191, 36.089, 48.715, 59.501, 65.759, 65.759, 59.501, 48.715, 36.089, 24.191, 14.673},
                                           {12.629, 20.822, 31.062, 41.930, 51.213, 56.599, 56.599, 51.213, 41.930, 31.062, 20.822, 12.629},
                                           {9.835, 16.216, 24.191, 32.655, 39.885, 44.080, 44.080, 39.885, 32.655, 24.191, 16.216, 9.835},
                                           {6.931, 11.427, 17.047, 23.012, 28.106, 31.062, 31.062, 28.106, 23.012, 17.047, 11.427, 6.931},
                                           {4.419, 7.286, 10.870, 14.673, 17.921, 19.806, 19.806, 17.921, 14.673, 10.870, 7.286, 4.419},
                                           {2.550, 4.204, 6.271, 8.465, 10.340, 11.427, 11.427, 10.340, 8.465, 6.271, 4.204, 2.550},
                                           {1.331, 2.195, 3.274, 4.419, 5.398, 5.966, 5.966, 5.398, 4.419, 3.274, 2.195, 1.331},
                                           {0.629, 1.037, 1.547, 2.088, 2.550, 2.818, 2.818, 2.550, 2.088, 1.547, 1.037, 0.629},
                                           {0.269, 0.443, 0.661, 0.892, 1.090, 1.204, 1.204, 1.090, 0.892, 0.661, 0.443, 0.269},
                                           {0.104, 0.171, 0.256, 0.345, 0.421, 0.466, 0.466, 0.421, 0.345, 0.256, 0.171, 0.104}};

bool in_bounds(int x, int y)
{
    return (x >= 0 && x < cols && y >= 0 && y < rows);
}

bool is_opponent_score_cell(int x, int y, const std::string &player, const std::vector<int> &score_cols)
{
    if (player != "circle")
    {
        return (y == top_score_row) && (std::find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
    }
    else
    {
        return (y == bottom_score_row) && (std::find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
    }
}

bool is_our_score_cell(int x, int y, const std::string &player, const std::vector<int> &score_cols)
{
    if (player == "circle")
    {
        return (y == top_score_row) && (std::find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
    }
    else
    {
        return (y == bottom_score_row) && (std::find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
    }
}

vector<vector<int>> get_river_flow_destinations(const FastBoard &board, int rx, int ry, int sx, int sy, const string &player, const vector<int> &score_cols, bool river_push = false) // checked, works!!
{
    vector<vector<int>> destinations;
    set<pair<int, int>> visited;
    queue<pair<int, int>> q;
    q.push({rx, ry});

    uint8_t player_id = (player == "circle") ? 1 : 2;

    while (!q.empty())
    {
        auto [x, y] = q.front();
        q.pop();

        if (!in_bounds(x, y) || visited.count({x, y}))
            continue;
        visited.insert({x, y});

        const Piece &cell = board[y][x];
        const Piece *current_cell = &cell;
        if (river_push && x == rx && y == ry)
        {
            current_cell = &board[sy][sx];
        }

        if (current_cell->empty())
        {
            if (!is_opponent_score_cell(x, y, player, score_cols))
                destinations.push_back({x, y});
            continue;
        }

        if (!current_cell->is_river())
            continue;

        vector<pair<int, int>> dirs;
        if (current_cell->is_horizontal())
            dirs = {{1, 0}, {-1, 0}};
        else
            dirs = {{0, 1}, {0, -1}};

        for (auto [dx, dy] : dirs)
        {
            int nx = x + dx, ny = y + dy;
            while (in_bounds(nx, ny))
            {
                if (is_opponent_score_cell(nx, ny, player, score_cols))
                    break;

                const Piece &next_cell = board[ny][nx];
                if (next_cell.empty())
                {
                    destinations.push_back({nx, ny});
                    nx += dx;
                    ny += dy;
                    continue;
                }
                if (nx == sx && ny == sy)
                {
                    nx += dx;
                    ny += dy;
                    continue;
                }
                if (next_cell.is_river())
                {
                    q.push({nx, ny});
                    break;
                }
                break;
            }
        }
    }

    set<pair<int, int>> seen;
    vector<vector<int>> unique_destinations;
    for (const auto &d : destinations)
    {
        pair<int, int> p = {d[0], d[1]};
        if (!seen.count(p))
        {
            seen.insert(p);
            unique_destinations.push_back(d);
        }
    }

    return unique_destinations;
}

vector<Move> generate_all_moves(const FastBoard &board, const string &player, const vector<int> &score_cols) // Checked, Works!!
{
    vector<Move> all_moves;
    vector<pair<int, int>> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    uint8_t player_id = (player == "circle") ? 1 : 2;
    string opponent_player = (player == "circle") ? "square" : "circle";

    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            const Piece &piece = board[y][x];
            if (piece.empty() || piece.owner != player_id)
                continue;

            if (piece.is_stone())
            {
                for (auto [dx, dy] : directions)
                {
                    int nx = x + dx, ny = y + dy;
                    if (!in_bounds(nx, ny))
                        continue;
                    if (is_opponent_score_cell(nx, ny, player, score_cols))
                        continue;

                    const Piece &target = board[ny][nx];

                    if (target.empty()) // Move a stone
                    {
                        all_moves.push_back(Move{"move", {x, y}, {nx, ny}, {}, "horizontal"});
                    }
                    else if (target.is_river()) // Move stone into river
                    {
                        auto flow_dest = get_river_flow_destinations(board, nx, ny, x, y, player, score_cols);
                        for (auto &d : flow_dest)
                        {
                            all_moves.push_back(Move{"move", {x, y}, d, {}, "horizontal"});
                        }
                    }
                    else
                    {
                        // A stone occupies target - check push possibilities
                        int px = nx + dx, py = ny + dy;
                        if (!in_bounds(px, py))
                            continue;
                        if (target.owner != player_id) // opponent stone, push
                        {
                            if (board[py][px].empty() && !is_opponent_score_cell(px, py, opponent_player, score_cols))
                            {
                                string orientation = target.is_horizontal() ? "horizontal" : "vertical";
                                all_moves.push_back(Move{"push", {x, y}, {nx, ny}, {px, py}, orientation});
                            }
                        }
                        else
                        {
                            // My stone, push
                            if (board[py][px].empty() && !is_opponent_score_cell(px, py, player, score_cols))
                            {
                                string orientation = target.is_horizontal() ? "horizontal" : "vertical";
                                all_moves.push_back(Move{"push", {x, y}, {nx, ny}, {px, py}, orientation});
                            }
                        }
                    }
                }
                // flips in both orientations
                all_moves.push_back(Move{"flip", {x, y}, {x, y}, {}, "horizontal"});
                all_moves.push_back(Move{"flip", {x, y}, {x, y}, {}, "vertical"});
            }
            else if (piece.is_river())
            {
                string orientation = piece.is_horizontal() ? "horizontal" : "vertical";
                for (auto [dx, dy] : directions)
                {
                    int nx = x + dx, ny = y + dy;
                    if (!in_bounds(nx, ny))
                        continue;
                    if (is_opponent_score_cell(nx, ny, player, score_cols))
                        continue;

                    const Piece &target = board[ny][nx];
                    if (target.empty())
                    { // move a river
                        all_moves.push_back(Move{"move", {x, y}, {nx, ny}, {}, orientation});
                    }
                    else if (target.is_river())
                    { // move river via river
                        auto flow_dest = get_river_flow_destinations(board, nx, ny, x, y, player, score_cols);
                        for (auto &d : flow_dest)
                            all_moves.push_back(Move{"move", {x, y}, d, {}, orientation});
                    }
                    else
                    { // river pushing stone away
                        if (target.owner == player_id)
                        { // my stone, push
                            auto flow_dest = get_river_flow_destinations(board, nx, ny, x, y,
                                                                         player, score_cols, true);

                            for (auto &d : flow_dest)
                            {
                                if (!is_opponent_score_cell(d[0], d[1], player, score_cols))
                                    all_moves.push_back(Move{"push", {x, y}, {nx, ny}, d, "horizontal"});
                            }
                        }
                        else
                        { // opponent stone, push
                            auto flow_dest = get_river_flow_destinations(board, nx, ny, x, y,
                                                                         opponent_player, score_cols, true);
                            for (auto &d : flow_dest)
                            {
                                if (!is_opponent_score_cell(d[0], d[1], opponent_player, score_cols))
                                    all_moves.push_back(Move{"push", {x, y}, {nx, ny}, d, "horizontal"});
                            }
                        }
                    }
                }
                // river to stone flip
                all_moves.push_back(Move{"flip", {x, y}, {x, y}, {}, "horizontal"});

                // rotate
                string new_orientation = piece.is_horizontal() ? "vertical" : "horizontal";
                all_moves.push_back(Move{"rotate", {x, y}, {}, {}, new_orientation});
            }
        }
    }
    return all_moves;
}

float get_position_value(int y, int x, const string &player) // for distance values from matrix
{
    if (player == "circle")
    {
        return distance_values[y][x];
    }
    else
    {
        return distance_values[rows - 1 - y][x];
    }
}

// FastBoard version of compute_valid_targets
MoveTargets compute_valid_targets(const FastBoard &board, int sx, int sy, const string &player, const vector<int> &score_cols) // Checked, Works!!
{
    // where a piece can go, via move or push
    MoveTargets result;

    if (!in_bounds(sx, sy) || board[sy][sx].empty())
    {
        return result;
    }

    uint8_t player_id = (player == "circle") ? 1 : 2;
    const Piece &piece = board[sy][sx];

    if (piece.owner != player_id)
    {
        return result;
    }

    vector<pair<int, int>> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for (auto [dx, dy] : directions)
    {
        int tx = sx + dx, ty = sy + dy;
        if (!in_bounds(tx, ty))
            continue;
        if (is_opponent_score_cell(tx, ty, player, score_cols))
            continue;

        const Piece &target = board[ty][tx];

        if (target.empty()) // move a piece to empty cell
        {
            result.moves.push_back({tx, ty});
        }
        else if (target.is_river()) // move piece via river to valid destinations
        {
            auto flow_dest = get_river_flow_destinations(board, tx, ty, sx, sy, player, score_cols);
            for (auto &d : flow_dest)
            {
                result.moves.push_back({d[0], d[1]});
            }
        }
        else // target is stone - Push
        {
            // Stone occupied - check push possibilities
            if (piece.is_stone()) // stone pushing stone
            {
                int px = tx + dx, py = ty + dy;
                if (in_bounds(px, py) && board[py][px].empty() &&
                    !is_opponent_score_cell(px, py, player, score_cols))
                {
                    result.pushes.push_back({{tx, ty}, {px, py}});
                }
            }
            else
            {
                // River pushing stone
                string pushed_player = (target.owner == 1) ? "circle" : "square";
                auto flow_dest = get_river_flow_destinations(board, tx, ty, sx, sy, pushed_player, score_cols, true);
                for (auto &d : flow_dest)
                {
                    if (!is_opponent_score_cell(d[0], d[1], player, score_cols))
                    {
                        result.pushes.push_back({{tx, ty}, {d[0], d[1]}});
                    }
                }
            }
        }
    }

    return result;
}

float calculate_piece_mobility_potential(const FastBoard &board, int x, int y, const string &player, const vector<int> &score_cols) // checked, works!!
{
    MoveTargets targets = compute_valid_targets(board, x, y, player, score_cols);
    float potential = 0.0f;

    // Calculate potential value based on reachable cells
    for (auto [tx, ty] : targets.moves)
    {
        float cell_value = get_position_value(ty, tx, player);
        potential += cell_value;

        // Bonus for reaching scoring areas
        if (is_our_score_cell(tx, ty, player, score_cols))
        {
            potential += 500.0f; // High bonus for direct scoring potential
        }
    }

    // Push moves also contribute potential (but less than direct moves)
    for (auto [target_pos, pushed_pos] : targets.pushes)
    {
        float cell_value = get_position_value(pushed_pos.second, pushed_pos.first, player);
        potential += cell_value; // * 0.7f;
        if (is_our_score_cell(pushed_pos.first, pushed_pos.second, player, score_cols))
        {
            if (!is_our_score_cell(target_pos.first, target_pos.second, player, score_cols))
                potential += 350.0f; // Scoring via push
            else
                potential += 150.0f; // Pushing within scoring area to create space
        }
    }

    return potential;
}

int detect_forcing_sequences(const FastBoard &board, const string &player, const vector<int> &score_cols) // checked, improved, works!!
{
    int forcing_moves = 0;
    uint8_t my_id = (player == "circle") ? 1 : 2;
    int my_score_row = (player == "circle") ? top_score_row : bottom_score_row;

    // Check each of my stones for 2-3 move winning sequences
    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            const Piece &piece = board[y][x];
            if (piece.owner != my_id)
                continue;

            MoveTargets targets = compute_valid_targets(board, x, y, player, score_cols);

            // Check direct moves first
            for (auto [tx, ty] : targets.moves)
            {
                if (ty == my_score_row && find(score_cols.begin(), score_cols.end(), tx) != score_cols.end())
                {
                    if (piece.is_stone())
                        forcing_moves += 30; // Direct winning move
                    else
                        forcing_moves += 20; // River to stone flip into scoring
                }

                // Check if this move sets up a win next turn
                FastBoard temp_board = board;
                vector<SavedPiece> saved;
                Move test_move = {"move", {x, y}, {tx, ty}, {}, piece.orientation == 0 ? "horizontal" : "vertical"};
                make_move(temp_board, test_move, saved);

                // After this move, can I score immediately?
                MoveTargets next_targets = compute_valid_targets(temp_board, tx, ty, player, score_cols);
                for (auto [ntx, nty] : next_targets.moves)
                {
                    if (nty == my_score_row && find(score_cols.begin(), score_cols.end(), ntx) != score_cols.end())
                    {
                        if (piece.is_stone())
                            forcing_moves += 5; // 2-move win sequence
                        else
                            forcing_moves += 3; // 2-move win sequence via river
                    }
                }

                undo_move(temp_board, saved);
            }

            // Check push moves
            for (auto [target_pos, pushed_pos] : targets.pushes)
            {
                if (is_our_score_cell(pushed_pos.second, my_score_row, player, score_cols))
                {
                    if (is_our_score_cell(target_pos.first, target_pos.second, player, score_cols))
                        forcing_moves += 7; // Pushing within scoring area to create space
                    else
                        forcing_moves += 12; // Direct winning push
                }

                // Check if this push sets up a win next turn
                FastBoard temp_board = board;
                vector<SavedPiece> saved;
                Move test_move = {"push", {x, y}, {target_pos.first, target_pos.second}, {pushed_pos.first, pushed_pos.second}, piece.orientation == 0 ? "horizontal" : "vertical"};
                make_move(temp_board, test_move, saved);
                MoveTargets next_targets = compute_valid_targets(temp_board, pushed_pos.first, pushed_pos.second, player, score_cols);

                for (auto [ntx, nty] : next_targets.moves)
                {
                    if (nty == my_score_row && find(score_cols.begin(), score_cols.end(), ntx) != score_cols.end())
                    {
                        if (piece.is_stone())
                            forcing_moves += 5; // 2-move win sequence via push
                        else
                            forcing_moves += 2; // 2-move win sequence via river , flip later
                    }
                }
                undo_move(temp_board, saved);
            }
        }
    }

    return forcing_moves;
}

float evaluate_zero_sum(const FastBoard &board, const string &player, const vector<int> &score_cols, const EvalWeights &weights)
{
    EvalFeatures my_features, opp_features;

    uint8_t my_id = (player == "circle") ? 1 : 2;
    uint8_t opp_id = (player == "circle") ? 2 : 1;
    string opponent = (player == "circle") ? "square" : "circle";

    // Define scoring zones
    int my_score_row = (player == "circle") ? top_score_row : bottom_score_row;
    int opp_score_row = (player == "circle") ? bottom_score_row : top_score_row;

    // stones in goal
    for (int x : score_cols)
    {
        if (board[my_score_row][x].owner == my_id)
        {
            if (board[my_score_row][x].is_stone())
                my_features.stones_in_goal += 2.0f; // stone worth double
            else
                my_features.stones_in_goal += 1.0f;
        }
        if (board[opp_score_row][x].owner == opp_id)
        {
            if (board[opp_score_row][x].is_stone())
                opp_features.stones_in_goal += 2.0f;
            else
                opp_features.stones_in_goal += 1.0f;
        }
    }

    // immediate threats
    my_features.immediate_threats = detect_forcing_sequences(board, player, score_cols);
    opp_features.immediate_threats = detect_forcing_sequences(board, opponent, score_cols);

    // Pieces that can score immediately
    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            const Piece &piece = board[y][x];
            if (piece.empty())
                continue;

            if (piece.owner == my_id && piece.is_stone())
            {
                MoveTargets targets = compute_valid_targets(board, x, y, player, score_cols);
                for (auto [tx, ty] : targets.moves)
                {
                    if (ty == my_score_row && find(score_cols.begin(), score_cols.end(), tx) != score_cols.end())
                    {
                        my_features.one_move_away += 1.0f;
                    }
                }
            }

            if (piece.owner == opp_id && piece.is_stone())
            {
                MoveTargets targets = compute_valid_targets(board, x, y, opponent, score_cols);
                for (auto [tx, ty] : targets.moves)
                {
                    if (ty == opp_score_row && find(score_cols.begin(), score_cols.end(), tx) != score_cols.end())
                    {
                        opp_features.one_move_away += 1.0f;
                    }
                }
            }
        }
    }

    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            const Piece &piece = board[y][x];
            if (piece.empty())
                continue;

            if (piece.owner == my_id && piece.is_stone())
            {
                int distance_to_goal = abs(y - my_score_row);
                float advancement_value = (10.0f - distance_to_goal) * 1.5f;

                // Bonus for stones in scoring columns
                if (find(score_cols.begin(), score_cols.end(), x) != score_cols.end())
                {
                    advancement_value += 30.0f;
                }

                // Bonus for stones close to goal
                if (distance_to_goal <= 3)
                {
                    advancement_value += 10.0f;
                }

                my_features.advancement_sum += advancement_value;

                // Penalty for stones too far from goal
                if (distance_to_goal > 8)
                {
                    my_features.advancement_sum -= 3.0f;
                }
            }

            if (piece.owner == opp_id && piece.is_stone())
            {
                int distance_to_goal = abs(y - opp_score_row);
                float advancement_value = (13.0f - distance_to_goal) * 1.5f;

                if (find(score_cols.begin(), score_cols.end(), x) != score_cols.end())
                {
                    advancement_value += 30.0f;
                }

                if (distance_to_goal <= 3)
                {
                    advancement_value += 10.0f;
                }

                opp_features.advancement_sum += advancement_value;
            }
        }
    }

    // centre control
    for (int y = 3; y <= 9; y++)
    {
        for (int x = 4; x <= 8; x++)
        {
            const Piece &piece = board[y][x];
            if (!piece.empty())
            {
                if (piece.owner == my_id)
                    my_features.center_control += 1.0f;
                else
                    opp_features.center_control += 1.0f;
            }
        }
    }

    // river support
    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            const Piece &piece = board[y][x];
            if (piece.empty() || piece.is_stone())
                continue;

            if (piece.owner == my_id)
            {
                float river_value = 2.0f;
                int distance_to_goal = abs(y - my_score_row);

                if (distance_to_goal <= 4)
                {
                    river_value += 3.0f;
                }

                if (find(score_cols.begin(), score_cols.end(), x) != score_cols.end())
                {
                    river_value += 8.0f;
                }

                // Check if river helps stones advance
                bool helps_stones = false;
                for (int sy = 0; sy < rows && !helps_stones; sy++)
                {
                    for (int sx = 0; sx < cols && !helps_stones; sx++)
                    {
                        const Piece &stone = board[sy][sx];
                        if (stone.owner == my_id && stone.is_stone())
                        {
                            MoveTargets targets = compute_valid_targets(board, sx, sy, player, score_cols);
                            for (auto [tx, ty] : targets.moves)
                            {
                                if (abs(tx - x) + abs(ty - y) <= 3 && abs(ty - my_score_row) < abs(sy - my_score_row))
                                {
                                    helps_stones = true;
                                    break;
                                }
                            }
                        }
                    }
                }

                if (helps_stones)
                {
                    river_value += 4.0f;
                }

                my_features.river_support += river_value;
            }

            if (piece.owner == opp_id)
            {
                float river_value = 2.0f;
                int distance_to_goal = abs(y - opp_score_row);
                if (distance_to_goal <= 4)
                    river_value += 3.0f;
                if (find(score_cols.begin(), score_cols.end(), x) != score_cols.end())
                {
                    river_value += 8.0f;
                }
                opp_features.river_support += river_value;
            }
        }
    }

    // piece counting and movement
    int my_stones = 0, opp_stones = 0;
    int my_rivers = 0, opp_rivers = 0;

    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            const Piece &piece = board[y][x];
            if (piece.empty())
                continue;

            if (piece.owner == my_id)
            {
                if (piece.is_stone())
                    my_stones++;
                else
                    my_rivers++;
            }
            if (piece.owner == opp_id)
            {
                if (piece.is_stone())
                    opp_stones++;
                else
                    opp_rivers++;
            }
        }
    }

    my_features.mobility = my_stones * 6.0f + my_rivers * 2.5f;
    opp_features.mobility = opp_stones * 6.0f + opp_rivers * 2.5f;

    // Penalty for too many rivers vs stones
    if (my_rivers > my_stones + 2)
    {
        my_features.mobility -= 50.0f;
    }
    if (opp_rivers > opp_stones + 2)
    {
        opp_features.mobility -= 20.0f;
    }

    // blocking
    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            const Piece &piece = board[y][x];
            if (piece.empty())
                continue;

            if (piece.owner == my_id)
            {
                if (is_opponent_score_cell(x, y - 1, opponent, score_cols) ||
                    is_opponent_score_cell(x, y + 1, opponent, score_cols) || (y == opp_score_row && (x == 3 || x == 8)))
                {
                    my_features.blocking += 1.0f; // 10 max
                }
            }

            if (piece.owner == opp_id)
            {
                if (is_opponent_score_cell(x, y - 1, player, score_cols) ||
                    is_opponent_score_cell(x, y + 1, player, score_cols) || (y == my_score_row && (x == 3 || x == 8)))
                {
                    opp_features.blocking += 1.0f;
                }
            }
        }
    }
    my_features.blocking *= (my_features.blocking <= 3) ? -1 : 1.0;
    opp_features.blocking *= (opp_features.blocking <= 3) ? -1 : 1.0;
    // weighted scored
    float my_score = weights.w_stones_goal * my_features.stones_in_goal +
                     weights.w_threats * my_features.immediate_threats +
                     weights.w_one_away * my_features.one_move_away +
                     weights.w_advancement * my_features.advancement_sum +
                     weights.w_center * my_features.center_control +
                     weights.w_river * my_features.river_support +
                     weights.w_mobility * my_features.mobility +
                     weights.w_blocking * my_features.blocking;
    // float opp_score = 0;
    float opp_score = weights.w_stones_goal * opp_features.stones_in_goal +
                      weights.w_threats * opp_features.immediate_threats +
                      weights.w_one_away * opp_features.one_move_away +
                      weights.w_advancement * opp_features.advancement_sum +
                      weights.w_center * opp_features.center_control +
                      weights.w_river * opp_features.river_support +
                      weights.w_mobility * opp_features.mobility +
                      weights.w_blocking * opp_features.blocking;

    // Emergency conditions (matching original logic)
    if (opp_features.stones_in_goal >= 3.0f)
    {
        my_score -= 5000.0f; // Emergency defense mode
    }
    if (my_features.stones_in_goal >= 2.0f)
    {
        my_score += 3000.0f; // Victory is near
    }

    return my_score - opp_score; // Zero sum
}

void make_move(FastBoard &board, const Move &move, std::vector<SavedPiece> &saved_pieces)
{
    saved_pieces.clear();

    auto save_piece = [&](int x, int y)
    {
        saved_pieces.push_back({{x, y}, board[y][x]});
    };

    if (move.action == "move")
    {
        int fx = move.from[0], fy = move.from[1];
        int tx = move.to[0], ty = move.to[1];

        save_piece(tx, ty);
        save_piece(fx, fy);

        if (board[ty][tx].empty())
        {
            board[ty][tx] = board[fy][fx];
            board[fy][fx] = {0, 0, 0};
        }
        else if (!move.pushed_to.empty())
        {
            int px = move.pushed_to[0], py = move.pushed_to[1];
            save_piece(px, py);

            board[py][px] = board[ty][tx];
            board[ty][tx] = board[fy][fx];
            board[fy][fx] = {0, 0, 0};
        }
    }
    else if (move.action == "push")
    {
        int fx = move.from[0], fy = move.from[1];
        int tx = move.to[0], ty = move.to[1];
        int px = move.pushed_to[0], py = move.pushed_to[1];

        save_piece(px, py);
        save_piece(tx, ty);
        save_piece(fx, fy);

        board[py][px] = board[ty][tx];
        board[ty][tx] = board[fy][fx];
        board[fy][fx] = {0, 0, 0};

        if (board[ty][tx].is_river())
        {
            board[ty][tx].side = 0;        // stone
            board[ty][tx].orientation = 0; // horizontal
        }
    }
    else if (move.action == "flip")
    {
        int fx = move.from[0], fy = move.from[1];
        save_piece(fx, fy);

        Piece &piece = board[fy][fx];
        if (piece.is_stone())
        {
            piece.side = 1; // river
            piece.orientation = (move.orientation == "horizontal") ? 0 : 1;
        }
        else
        {
            piece.side = 0;        // stone
            piece.orientation = 0; // horizontal
        }
    }
    else if (move.action == "rotate")
    {
        int fx = move.from[0], fy = move.from[1];
        save_piece(fx, fy);

        Piece &piece = board[fy][fx];
        if (piece.is_river())
        {
            piece.orientation = piece.is_horizontal() ? 1 : 0;
        }
    }
}

void undo_move(FastBoard &board, const std::vector<SavedPiece> &saved_pieces)
{
    for (const auto &saved : saved_pieces)
    {
        int x = saved.pos[0], y = saved.pos[1];
        board[y][x] = saved.data;
    }
}

float evaluate_move(FastBoard &board, const Move &move, const string &player, const vector<int> &score_cols, const EvalWeights &weights = EvalWeights())
{
    vector<SavedPiece> saved_pieces;
    make_move(board, move, saved_pieces);
    float score = evaluate_zero_sum(board, player, score_cols, weights);
    undo_move(board, saved_pieces);
    return score;
}

void sort_moves(vector<Move> &moves, FastBoard &board, const string &player, const vector<int> &score_cols)
{
    vector<ScoredMove> scored_moves;

    for (const auto &move : moves)
    {
        ScoredMove sm;
        sm.move = move;
        sm.score = evaluate_move(board, move, player, score_cols);
        scored_moves.push_back(sm);
    }

    sort(scored_moves.begin(), scored_moves.end(),
         [](const ScoredMove &a, const ScoredMove &b)
         {
             return a.score > b.score;
         });

    moves.clear();
    for (const auto &sm : scored_moves)
    {
        moves.push_back(sm.move);
        if (sm.score >= 100000)
        {
            break; // Found winning move, no need to evaluate others
        }
    }
}

float alpha_beta(FastBoard &board, const string &player, const vector<int> &score_cols, int depth, float alpha, float beta, bool maximizingPlayer, const string &my_side, const EvalWeights &weights = EvalWeights())
{
    if (depth == 0)
    {
        return evaluate_zero_sum(board, my_side, score_cols, weights);
    }

    vector<Move> moves = generate_all_moves(board, player, score_cols);
    if (moves.empty())
    {
        return maximizingPlayer ? -10000.0f : 10000.0f;
    }

    sort_moves(moves, board, player, score_cols);

    if (moves.size() > 100) // Reduced from 100
    {
        moves.resize(100);
    }

    string opponent_player = (player == "circle") ? "square" : "circle";

    if (maximizingPlayer)
    {
        float maxEval = -1e9;
        vector<SavedPiece> saved_pieces;

        for (auto &move : moves)
        {
            make_move(board, move, saved_pieces);
            float eval = alpha_beta(board, opponent_player, score_cols, depth - 1, alpha, beta, false, my_side, weights);
            undo_move(board, saved_pieces);

            maxEval = max(maxEval, eval);
            alpha = max(alpha, eval);
            if (beta <= alpha)
                break;
        }
        return maxEval;
    }
    else
    {
        float minEval = 1e9;
        vector<SavedPiece> saved_pieces;

        for (auto &move : moves)
        {
            make_move(board, move, saved_pieces);
            float eval = alpha_beta(board, opponent_player, score_cols, depth - 1, alpha, beta, true, my_side, weights);
            undo_move(board, saved_pieces);

            minEval = min(minEval, eval);
            beta = min(beta, eval);
            if (beta <= alpha)
                break;
        }
        return minEval;
    }
}

Move flip_river_in_area(FastBoard &board, const string &side)
{
    Move flip;
    for (int x = 4; x < 8; x++)
    {
        int y = (side == "circle") ? top_score_row : bottom_score_row;
        if (!board[y][x].empty() && board[y][x].is_river())
        {
            return {"flip", {x, y}, {x, y}, {}, "horizontal"};
        }
    }

    return {"", {0, 0}, {0, 0}, {}, ""};
}

class StudentAgent
{
    int move_count = 0;
    float calculate_time_limit(int move_count, float base_time, float max_time = 5.0f)
    {
        float progress = (float)move_count / 450.0f;
        float time_multiplier = 1.0f + sqrt(progress) * 1.5f; // 1.0 to 2.5 range
        return min(base_time * time_multiplier, max_time);
    }

public:
    explicit StudentAgent(std::string side) : side(std::move(side)), gen(rd())
    {
    }

    Move choose(const std::vector<std::vector<std::map<std::string, std::string>>> &board, const std::vector<int> &score_cols, float current_player_time = 10.0, float opponent_time = 10.0)
    {

        auto start_time = chrono::high_resolution_clock::now();
        float time_limit = calculate_time_limit(move_count++, current_player_time / 50.0f);

        FastBoard fast_board = convert_to_fast_board(board);

        std::vector<Move> all_moves = generate_all_moves(fast_board, side, score_cols);

        Move flip = flip_river_in_area(fast_board, side);
        if (!flip.action.empty())
            return flip;

        if (all_moves.empty())
        {
            return {"move", {0, 0}, {0, 0}, {}, "horizontal"};
        }

        if (all_moves.size() > 100)
        {
            all_moves.resize(100);
        }

        std::vector<ScoredMove> scored_moves;

        for (const auto &move : all_moves)
        {
            scored_moves.push_back({move, evaluate_move(fast_board, move, side, score_cols)});
        }
        std::sort(scored_moves.begin(), scored_moves.end(), [](const ScoredMove &a, const ScoredMove &b)
                  { return a.score > b.score; });

        if (scored_moves.size() > 30)
        {
            scored_moves.resize(30);
        }

        Move best_move = scored_moves[0].move; // first move is greedy
        if (move_count <= 7)
            return best_move; // play greedy for first 7 moves, to save time for later
        float best_score = -1e9;

        // Sort by initial heuristic scores

        int depth = 2;
        for (depth = 2; depth <= 10; ++depth)
        {
            auto current_time = chrono::high_resolution_clock::now();
            float elapsed = chrono::duration<float>(current_time - start_time).count();

            if (elapsed > time_limit * 0.8f)
            {
                break;
            }

            // new scored_moves for this depth
            std::vector<ScoredMove> current_depth_moves;

            // Process moves in order from previous depth (best first)
            for (auto &scored_move : scored_moves)
            {
                current_time = chrono::high_resolution_clock::now();
                elapsed = chrono::duration<float>(current_time - start_time).count();
                if (elapsed > time_limit * 0.9f)
                {
                    break;
                }

                vector<SavedPiece> saved_pieces;
                make_move(fast_board, scored_move.move, saved_pieces);

                float score = alpha_beta(fast_board, opponent(side), score_cols,
                                         depth - 1, -1e9, 1e9, false, side);

                undo_move(fast_board, saved_pieces);

                // Store the updated score for this move
                current_depth_moves.push_back({scored_move.move, score});

                if (score > best_score)
                {
                    best_score = score;
                    best_move = scored_move.move;
                }
            }

            // Sort moves by their minimax scores for next depth
            std::sort(current_depth_moves.begin(), current_depth_moves.end(),
                      [](const ScoredMove &a, const ScoredMove &b)
                      {
                          return a.score > b.score;
                      });

            // Update scored_moves for next iteration
            scored_moves = std::move(current_depth_moves);
        }
        return best_move;
    }

private:
    std::string side;
    std::random_device rd;
    std::mt19937 gen;
};

PYBIND11_MODULE(student_agent_module, m)
{
    py::class_<Move>(m, "Move")
        .def_readonly("action", &Move::action)
        .def_readonly("from_pos", &Move::from)
        .def_readonly("to_pos", &Move::to)
        .def_readonly("pushed_to", &Move::pushed_to)
        .def_readonly("orientation", &Move::orientation);

    py::class_<StudentAgent>(m, "StudentAgent")
        .def(py::init<std::string>())
        .def("choose", &StudentAgent::choose);
}