// 2.62     Nov 10, 05  hyperparameter autosearch: observations weighted by inverse stddev
// 3.02     Sep 01, 11  watch for denormal or infinite limits during autosearch
// MAS      Jan 02, 15  changed first step direction if hyperparameter is very small

#define  _USE_MATH_DEFINES
#include <cmath>
#include <stdexcept>
#include <limits>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/LU>

//#include "tnt_array2d.h"
//#include "tnt_array2d_utils.h"
//#include "jama_lu.h"
//
//#include "logging.h"

#include "HParSearch.h"

//observations weighted by inverse stddev
QuadrCoefs QuadrLogFit( const map<double,UniModalSearch::MS> & y_by_x )
{
    unsigned n = y_by_x.size();

    //prepare for case weights: inverse of stddev; beware of zero stddev
    double minstddev=numeric_limits<double>::max();
    for (map<double,UniModalSearch::MS>::const_iterator itr=y_by_x.begin(); itr!=y_by_x.end(); itr++) {
        if (0 < itr->second.s && itr->second.s < minstddev) {
        	minstddev = itr->second.s;
        }
    }
    const double zeroadjust = minstddev<numeric_limits<double>::max() ? 100.0/minstddev //non-zeroes present
        : 1.0; //all weights will be equal

    Eigen::MatrixXd X( n, 3 ); //nRows, nCols
    Eigen::MatrixXd XTW( 3, n ); //nRows, nCols
    Eigen::VectorXd Y( n );
    int i = 0;
    for (map<double,UniModalSearch::MS>::const_iterator itr=y_by_x.begin(); itr!=y_by_x.end();
        itr++, i++) {
        double weight = itr->second.s>0 ? 1/itr->second.s : zeroadjust;
        X(i,0) = XTW(0,i) = 1.0;
        X(i,1) = XTW(1,i) = log( itr->first );
        X(i,2) = XTW(2,i) = log( itr->first ) * log( itr->first );
        for (int j=0; j < 3; j++) { //for no weighting, just skip this
            XTW(j,i) *= weight;
        }
        Y(i) = itr->second.m;
    }

    Eigen::MatrixXd XTX =  XTW * X;
    Eigen::VectorXd XTY = XTW * Y;
    Eigen::VectorXd b_hat = XTX.fullPivLu().solve(XTY);

    // TODO Check numerical accuracy and throw error

    QuadrCoefs ret;
    ret.c0 = b_hat(0);
    ret.c1 = b_hat(1);
    ret.c2 = b_hat(2);
    return ret;
}

StepValue UniModalSearch::step() //recommend: do/not next step, and the next x value
{
    StepValue ret(true,0,0);
    switch( y_by_x.size() ) {
    case 0: ret.second = 1; break;
    case 1:
        if (y_by_x.begin()->first < m_first_cut)
            ret.second = y_by_x.begin()->first * m_stdstep;
        else
            ret.second = y_by_x.begin()->first / m_stdstep;
            //we divide here because step with more penalty is safer numerically
        break;
    case 2:
        if( y_by_x.begin()->second.m > y_by_x.rbegin()->second.m )
            ret.second = y_by_x.begin()->first / m_stdstep;
        else
            ret.second = y_by_x.rbegin()->first * m_stdstep;
        break;
    default: // 3 or more
		if( y_by_x.begin()->first==best->first ) {  //max is at the left - move to the left
            ret.second = best->first / m_stdstep;
			if( !(best->first > numeric_limits<double>::denorm_min()) ) { //inf or nan
				ret.first = false;
			    ret.expected = best->second.m;
			}
		}else if( y_by_x.rbegin()->first==best->first ) { //max is at the right - move to the right
            ret.second = best->first * m_stdstep;
			if( !(best->first < numeric_limits<double>::infinity()) ) { //inf or nan
				ret.first = false;
			    ret.expected = best->second.m;
			}
		}else { //max is 'bracketed'
            QuadrCoefs coefs = QuadrLogFit( y_by_x );
            double log_argmax = - coefs.c1 / coefs.c2 / 2;
            double expected_max = - coefs.c1*coefs.c1 / coefs.c2 / 4 + coefs.c0;

			double maxval = best->second.m;
            if( 0==maxval )  ret.first=false;
            else if( (expected_max-maxval)/fabs(maxval) < m_stop_by_y )  ret.first=false;
            else if( fabs(log_argmax-log(best->first)) < m_stop_by_x )  ret.first=false;
            //double deriv = 2*coefs.c2*best->first + coefs.c1;    cout<<"\nderiv  "<<deriv;
            //ret.first = fabs(deriv) > stop_by_y;

            ret.second = exp( log_argmax );
            ret.expected = expected_max;
            //map<double,double>::const_iterator left = best; left--;
            //map<double,double>::const_iterator right = best; right++;
//            std::cout
//            //Log(6)
//            <<"\nSearch step "<<ret.second<<" stop_by_y "<<((expected_max-maxval)/fabs(maxval))
//                <<" stop_by_x "<<(fabs(log_argmax-log(best->first))) << endl;
        }
    }
    return ret;
}

//void UniModalSearch::dump(std::ostream& stream) const {
//    int i = 0;
//    for (map<double,UniModalSearch::MS>::const_iterator itr=y_by_x.begin(); itr!=y_by_x.end();
//        itr++, i++) {
//    	stream << "search[ " << itr->first << " ] = " << itr->second << std::endl;
//    }
//}


std::ostream& operator<< (std::ostream& stream, const UniModalSearch& search) {
//	search.dump(stream);

    for (map<double,UniModalSearch::MS>::const_iterator itr=search.y_by_x.begin(); itr!=search.y_by_x.end();
        itr++) {
    	stream << "search[ " << itr->first << " ] = " << itr->second.m
    			<< "(" << itr->second.s << ")" << std::endl;
    }

	return stream;
}

#ifdef UNITTEST_
#include <iostream>
void main() {
    map<double,double> y_by_x;
    //example llkl.d001145.b prepared in R
    y_by_x[ 2.000000e+04 ] = -408.199;
    y_by_x[ 2.002884e+03 ] = -328.581;
    y_by_x[ 2.000000e+02 ] = -281.788;
    y_by_x[ 2.002884e+01 ] = -229.966;
    y_by_x[ 2.000000e+00 ] = -184.908;
    y_by_x[ 2.002884e-01 ] = -150.792;
    y_by_x[ 2.000000e-02 ] = -132.081;
    y_by_x[ 2.002884e-03 ] = -127.708;
    y_by_x[ 2.000000e-04 ] = -146.397;
    y_by_x[ 2.002884e-05 ] = -183.208;
    QuadrCoefs c = QuadrLogFit( y_by_x );
    //R results:
    // (Intercept)     logvar2      logvar
    // -171.030466   -1.186039  -12.662184
    cout<<"c0="<<c.c0<<"  c1="<<c.c1<<"  c2="<<c.c2<<endl;

    UniModalSearch s;
    for( map<double,double>::const_iterator itr=y_by_x.begin(); itr!=y_by_x.end(); itr++ )
        s.tried( itr->first, itr->second );
    //R result: 0.004805411
    cout<<"step="<<s.step().second<<endl;
}
#endif //UNITTEST_


/*
    Copyright (c) 2002, 2003, 2004, 2005, 2006, 2007, Rutgers University, New Brunswick, NJ, USA.

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
    BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
    ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    Except as contained in this notice, the name(s) of the above
    copyright holders, DIMACS, and the software authors shall not be used
    in advertising or otherwise to promote the sale, use or other
    dealings in this Software without prior written authorization.
*/
