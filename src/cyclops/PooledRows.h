/*
 * PooledRows.h
 *
 * Person-interval rows of pooled logistic regression ("plr"). A row of the data (of X, y and the
 * weights) is a subject; subject i has rows [rowStart[i], rowStart[i + 1]), one per interval at
 * risk. Explicit rows store each row's interval and outcome; implicit rows store only the first
 * interval of each subject, its rows being consecutive intervals, with the event (y of the subject)
 * in the last. The link is the logit or the complementary log-log. Time-basis columns and time x covariate interactions multiply a column's values in a
 * row by the time basis at the row's interval.
 */

#ifndef POOLEDROWS_H_
#define POOLEDROWS_H_

#include <cstdint>
#include <cstddef>
#include <utility>
#include <vector>

namespace bsccs {

struct PooledRows {
    bool loaded = false;
    bool implicitRows = false;
    bool cloglog = false;               // link: complementary log-log (else logit)
    std::vector<size_t> rowStart;       // subjects + 1
    std::vector<int> interval;          // explicit: 0-based interval of each row
    std::vector<char> event;            // explicit: outcome of each row
    std::vector<int> entry;             // implicit: 0-based first interval of each subject
    std::vector<double> rowWeight;      // empty, or one per row
    std::vector<double> rowOffset;      // empty, or one per row
    std::vector<double> subjectOffset;  // empty, or one per subject
    int intervals = 0;                  // rows of the time basis
    int basisColumns = 0;               // columns of the time basis
    std::vector<double> basis;          // intervals x basisColumns, by interval
    std::vector<std::pair<int64_t, int>> timeColumns; // covariate id -> its time-basis column

    size_t rows() const { return rowStart.empty() ? 0 : rowStart.back(); }

    int intervalOf(size_t i, size_t r) const {
        return implicitRows ? entry[i] + static_cast<int>(r - rowStart[i]) : interval[r];
    }

    bool eventOf(size_t i, size_t r, double y) const {
        return implicitRows ? (y != 0.0 && r + 1 == rowStart[i + 1]) : event[r] != 0;
    }
};

} // namespace bsccs

#endif /* POOLEDROWS_H_ */
