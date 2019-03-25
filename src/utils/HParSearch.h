#ifndef HYPER_PARAMETER_SEARCH_HPP_
#define HYPER_PARAMETER_SEARCH_HPP_

#include <map>
/*#include <ostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>*/
using namespace std;

struct QuadrCoefs {
    double c0,c1,c2;
};

struct StepValue {
    bool first;
    double second;
    double expected;

    StepValue(bool cont, double x, double y) : first(cont), second(x), expected(y) { }
};

/* UniModalSearch:
 - collects loglikelihood results from previous steps
 - suggests next step
*/
class UniModalSearch {
public:
    struct MS{ //mean and stddev
        double m; double s;
        MS( double m_=0, double s_=0 ) : m(m_), s(s_) {}
    };
    double bestx() const {
        return best->first; }
    double besty() const {
        return best->second.m; }
    void tried( double x, double y, double y_stddev=0.0 ) //get to know next result (x,y)
    {
        y_by_x[x] = MS(y,y_stddev);
        if( y_by_x.size()==1 ) { //this is the first
            best = y_by_x.begin();
        } else {
//            double improve = (y - best->second.m) / best->second.m;
            if( y > best->second.m )
                best = y_by_x.find( x );
        }
    }
    StepValue step(); // recommend: do/not next step, the next x value
    //ctor
    UniModalSearch( double stdstep=100, double stop_by_y=.01, double stop_by_x=log(1.5),
        double firstCut=1.0 )
        : m_stdstep(stdstep), m_stop_by_y(stop_by_y), m_stop_by_x(stop_by_x), m_first_cut(firstCut) {}

    friend std::ostream& operator<< (std::ostream& stream, const UniModalSearch& search);

    void dump(std::ostream& stream) const;

private:
    const double m_stdstep;
    const double m_stop_by_y, m_stop_by_x;
    const double m_first_cut;
    std::map<double,MS> y_by_x;
    std::map<double,MS>::const_iterator best;
};

QuadrCoefs QuadrLogFit( const map<double,UniModalSearch::MS> & y_by_x );

#endif //HYPER_PARAMETER_SEARCH_HPP_

/*
    Copyright 2005, Rutgers University, New Brunswick, NJ.

    All Rights Reserved

    Permission to use, copy, and modify this software and its documentation for any purpose
    other than its incorporation into a commercial product is hereby granted without fee,
    provided that the above copyright notice appears in all copies and that both that
    copyright notice and this permission notice appear in supporting documentation, and that
    the names of Rutgers University, DIMACS, and the authors not be used in advertising or
    publicity pertaining to distribution of the software without specific, written prior
    permission.

    RUTGERS UNIVERSITY, DIMACS, AND THE AUTHORS DISCLAIM ALL WARRANTIES WITH REGARD TO
    THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    ANY PARTICULAR PURPOSE. IN NO EVENT SHALL RUTGERS UNIVERSITY, DIMACS, OR THE AUTHORS
    BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
    RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
    NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
    PERFORMANCE OF THIS SOFTWARE.
*/
