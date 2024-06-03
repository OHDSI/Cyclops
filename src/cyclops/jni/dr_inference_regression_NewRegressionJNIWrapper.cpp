#include "dr_inference_regression_NewRegressionJNIWrapper.h"

// #include "ccd.h"
#include "Rcpp.h"
#include "CyclicCoordinateDescent.h"
#include "ModelData.h"
#include "io/InputReader.h"
#include "RcppCyclopsInterface.h"

using namespace bsccs;

extern "C"
JNIEXPORT jint JNICALL Java_dr_inference_regression_NewRegressionJNIWrapper_loadData
  (JNIEnv *env, jobject obj, jstring javaFileName) {
    return -1;
}

extern "C"
JNIEXPORT jdouble JNICALL Java_dr_inference_regression_NewRegressionJNIWrapper_getLogLikelihood
  (JNIEnv *env, jobject obj, jint instance) {
    return getInterface(instance)->getCcd().getLogLikelihood();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_regression_NewRegressionJNIWrapper_getLogLikelihoodGradient
  (JNIEnv *env, jobject obj, jint instance, jdoubleArray javaArray) {

    auto& ccd = getInterface(instance)->getCcd();

    jsize len = env->GetArrayLength(javaArray);
    jdouble* gradient = env->GetDoubleArrayElements(javaArray, NULL);
    len = std::min(static_cast<int>(len), ccd.getBetaSize());

    for (int index = 0; index < len; ++index) {
        gradient[index] = ccd.getLogLikelihoodGradient(index);
    }

    env->ReleaseDoubleArrayElements(javaArray, gradient, 0); // copy values back
}

extern "C"
JNIEXPORT jdouble JNICALL Java_dr_inference_regression_NewRegressionJNIWrapper_getLogPrior
  (JNIEnv *env, jobject obj, jint instance) {
	return getInterface(instance)->getCcd().getLogPrior();
}

extern "C"
JNIEXPORT jdouble JNICALL Java_dr_inference_regression_NewRegressionJNIWrapper_getBeta
  (JNIEnv *env, jobject obj, jint instance, jint index) {
 	return getInterface(instance)->getCcd().getBeta(index);
}

extern "C"
JNIEXPORT jdouble JNICALL Java_dr_inference_regression_NewRegressionJNIWrapper_getHessian
  (JNIEnv *env, jobject obj, jint instance, jint index1, jint index2) {
 	// return getInterface(instance)->getCcd().getAsymptoticPrecision(index1, index2);
 	return -1;
}

extern "C"
JNIEXPORT jint JNICALL Java_dr_inference_regression_NewRegressionJNIWrapper_getBetaSize
  (JNIEnv *env, jobject obj, jint instance) {
	return getInterface(instance)->getCcd().getBetaSize();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_regression_NewRegressionJNIWrapper_setBeta__IID
  (JNIEnv *env, jobject obj, jint instance, jint index, jdouble value) {
	getInterface(instance)->getCcd().setBeta(index, value);
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_regression_NewRegressionJNIWrapper_setBeta__I_3D
  (JNIEnv *env, jobject obj, jint instance, jdoubleArray inValues) {
	jsize size = env->GetArrayLength( inValues );
	std::vector<double> input( size );
	env->GetDoubleArrayRegion( inValues, 0, size, &input[0] );
	getInterface(instance)->getCcd().setBeta(input);
}

extern "C"
JNIEXPORT jdouble JNICALL Java_dr_inference_regression_NewRegressionJNIWrapper_getHyperprior
  (JNIEnv *env, jobject obj, jint index, jint instance) {
  	auto prior = getInterface(instance)->getCcd().getHyperprior();
	return prior[index];
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_regression_NewRegressionJNIWrapper_setHyperprior
  (JNIEnv *env, jobject obj, jint instance, jint index, jdouble value) {
	getInterface(instance)->getCcd().setHyperprior(index, value);
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_regression_NewRegressionJNIWrapper_findMode
  (JNIEnv *env, jobject obj, jint instance) {
    // auto& interface = getInterface(instance)->getCcd();
    // interface.fitModel();
    getInterface(instance)->fitModel();
	// fitModel(interface.ccd, interface.arguments));
}

extern "C"
JNIEXPORT jint JNICALL Java_dr_inference_regression_NewRegressionJNIWrapper_getUpdateCount
  (JNIEnv *env, jobject obj, jint instance) {
	return getInterface(instance)->getCcd().getUpdateCount();
}

extern "C"
JNIEXPORT jint JNICALL Java_dr_inference_regression_NewRegressionJNIWrapper_getLikelihoodCount
  (JNIEnv *env, jobject obj, jint instance) {
	return getInterface(instance)->getCcd().getLikelihoodCount();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_regression_NewRegressionJNIWrapper_setPriorType
  (JNIEnv *env, jobject obj, jint instance, jint type) {
 	getInterface(instance)->getCcd().setPriorType(type);
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_regression_NewRegressionJNIWrapper_makeDirty
  (JNIEnv *env, jobject obj, jint instance) {
  getInterface(instance)->getCcd().makeDirty();
}

