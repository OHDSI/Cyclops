#ifndef ITERATORS_H
#define ITERATORS_H

#include "CompressedIndicatorMatrix.h"

/**
 * Iterators for dense, sparse and indicator vectors.  Each can be passed as a
 * template parameter to functions for compile-time specialization and optimization.
 * If performance is not an issue, GenericIterator wraps all three formats into a
 * single run-time determined iterator
 */

// Iterator for a sparse of indicators column
class IndicatorIterator {
  public:

	typedef real Scalar;
	typedef int Index;

	inline IndicatorIterator(const CompressedIndicatorMatrix& mat, Index column)
	  : mIndices(mat.getCompressedColumnVector(column)),
	    mId(0), mEnd(mat.getNumberOfEntries(column)){
		// Do nothing
	}

	inline IndicatorIterator(const std::vector<int>& vec, Index max = 0)
	: mIndices(vec.data()), mId(0), mEnd(vec.size()) {
		// Do nothing
	}

//	inline IndicatorIterator(const std::vector<int>& vec, Index column, Index max)
//	: mIndices(vec.data()), mId(0), mEnd(vec.size()) {
//		// Do nothing
//	}

    inline IndicatorIterator& operator++() { ++mId; return *this; }
    inline const Scalar value() const { return static_cast<Scalar>(1); }

    inline Index index() const { return mIndices[mId]; }
    inline operator bool() const { return (mId < mEnd); }

  protected:
    const Index* mIndices;
    Index mId;
    const Index mEnd;
};

// Iterator for a sparse column
class SparseIterator {
  public:

	typedef real Scalar;
	typedef int Index;

	inline SparseIterator(const CompressedIndicatorMatrix& mat, Index column)
	  : mValues(mat.getDataVector(column)), mIndices(mat.getCompressedColumnVector(column)),
	    mId(0), mEnd(mat.getNumberOfEntries(column)){
		// Do nothing
	}

    inline SparseIterator& operator++() { ++mId; return *this; }

    inline const Scalar& value() const { return mValues[mId]; }
    inline Scalar& valueRef() { return const_cast<Scalar&>(mValues[mId]); }

    inline Index index() const { return mIndices[mId]; }
    inline operator bool() const { return (mId < mEnd); }

  protected:
    const Scalar* mValues;
    const Index* mIndices;
    Index mId;
    const Index mEnd;
};

// Counting iterator
class CountingIterator {
public:
	typedef int Index;

	inline CountingIterator(const std::vector<int>& vec, Index end)
	: mId(0), mEnd(end) {
		// Do nothing
	}

	inline CountingIterator& operator++() { ++mId; return *this; }
	inline Index index() const  { return mId; }
	inline operator bool() const { return (mId < mEnd); }

protected:
	Index mId;
	const Index mEnd;
};

// Iterator for a dense column
class DenseIterator {
  public:

	typedef real Scalar;
	typedef int Index;

	inline DenseIterator(const CompressedIndicatorMatrix& mat, Index column)
	  : mValues(mat.getDataVector(column)),
	    mId(0), mEnd(mat.getNumberOfRows()){
		// Do nothing
	}

    inline DenseIterator& operator++() { ++mId; return *this; }

    inline const Scalar& value() const { return mValues[mId]; }
    inline Scalar& valueRef() { return const_cast<Scalar&>(mValues[mId]); }

    inline Index index() const { return mId; }
    inline operator bool() const { return (mId < mEnd); }

  protected:
    const Scalar* mValues;
    Index mId;
    const Index mEnd;
};

// Generic iterator for a run-time format determined column
class GenericIterator {
  public:

	typedef real Scalar;
	typedef int Index;

	inline GenericIterator(const CompressedIndicatorMatrix& mat, Index column)
	  : mFormatType(mat.getFormatType(column)),
	    mId(0) {
		if (mFormatType == DENSE) {
			mValues = mat.getDataVector(column);
			mIndices = NULL;
			mEnd = mat.getNumberOfRows();
		} else {
			if (mFormatType == SPARSE) {
				mValues = mat.getDataVector(column);
			} else {
				mValues = NULL;
			}
			mIndices = mat.getCompressedColumnVector(column);
			mEnd = mat.getNumberOfEntries(column);
		}
	}

    inline GenericIterator& operator++() { ++mId; return *this; }

    inline const Scalar value() const {
    	if (mFormatType == INDICATOR) {
    		return static_cast<Scalar>(1);
    	} else {
    		return mValues[mId];
    	}
    }
//    inline Scalar& valueRef() { return const_cast<Scalar&>(mValues[mId]); }

    inline Index index() const {
    	if (mFormatType == DENSE) {
    		return mId;
    	} else {
    		return mIndices[mId];
    	}
    }
    inline operator bool() const { return (mId < mEnd); }

  protected:
    const FormatType mFormatType;
    Scalar* mValues;
    Index* mIndices;
    Index mId;
    Index mEnd;
};


#endif // ITERATORS_H
