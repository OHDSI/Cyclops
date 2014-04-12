#include "dr_inference_regression_RegressionJNIWrapper.h"

#include "ccd.h"
#include "CyclicCoordinateDescent.h"
#include "ModelData.h"
#include "io/InputReader.h"

using namespace bsccs;

struct RegressionModel {
	CyclicCoordinateDescent* ccd;
	AbstractModelSpecifics* model;
	ModelData* modelData;
	CCDArguments* arguments;
};
	
std::vector<RegressionModel> instances;

extern "C"
JNIEXPORT jint JNICALL Java_dr_inference_regression_RegressionJNIWrapper_loadData
  (JNIEnv *env, jobject obj, jstring javaFileName) {

	const char *nativeFileName = env->GetStringUTFChars(javaFileName, 0);
	fprintf(stderr,"%s\n", nativeFileName);

	RegressionModel rModel;
	rModel.arguments = new CCDArguments;
	setDefaultArguments(*(rModel.arguments));   
	        
     (rModel.arguments)->inFileName = nativeFileName;
     (rModel.arguments)->modelName = "sccs";
     (rModel.arguments)->fileFormat ="sccs";
     (rModel.arguments)->outFileName = "r_out.txt";
     (rModel.arguments)->noiseLevel = SILENT;

	double timeInitialize = initializeModel(&(rModel.modelData), &(rModel.ccd), &(rModel.model), *(rModel.arguments));


	instances.push_back(rModel);
	   
	    env->ReleaseStringUTFChars(javaFileName, nativeFileName);
	    	    
	return instances.size() - 1;
}

extern "C"
JNIEXPORT jdouble JNICALL Java_dr_inference_regression_RegressionJNIWrapper_getLogLikelihood
  (JNIEnv *env, jobject obj, jint instance) {
 	return instances[instance].ccd->getLogLikelihood(); 
}

extern "C"
JNIEXPORT jdouble JNICALL Java_dr_inference_regression_RegressionJNIWrapper_getLogPrior
  (JNIEnv *env, jobject obj, jint instance) {
	return instances[instance].ccd->getLogPrior();
}

extern "C"
JNIEXPORT jdouble JNICALL Java_dr_inference_regression_RegressionJNIWrapper_getBeta
  (JNIEnv *env, jobject obj, jint instance, jint index) {
 	return instances[instance].ccd->getBeta(index); 
}

extern "C"
JNIEXPORT jdouble JNICALL Java_dr_inference_regression_RegressionJNIWrapper_getHessian
  (JNIEnv *env, jobject obj, jint instance, jint index1, jint index2) {
 	return instances[instance].ccd->getAsymptoticPrecision(index1, index2);
}

extern "C"
JNIEXPORT jint JNICALL Java_dr_inference_regression_RegressionJNIWrapper_getBetaSize
  (JNIEnv *env, jobject obj, jint instance) {
	return instances[instance].ccd->getBetaSize();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_regression_RegressionJNIWrapper_setBeta__IID
  (JNIEnv *env, jobject obj, jint instance, jint index, jdouble value) {
	instances[instance].ccd->setBeta(index, value);
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_regression_RegressionJNIWrapper_setBeta__I_3D
  (JNIEnv *env, jobject obj, jint instance, jdoubleArray inValues) {
	jsize size = env->GetArrayLength( inValues );
	std::vector<double> input( size );
	env->GetDoubleArrayRegion( inValues, 0, size, &input[0] );
	instances[instance].ccd->setBeta(input);

//	fprintf(stderr,"Not yet implemented (setBeta - array).\n");
//	exit(-1);
}

extern "C"
JNIEXPORT jdouble JNICALL Java_dr_inference_regression_RegressionJNIWrapper_getHyperprior
  (JNIEnv *env, jobject obj, jint instance) {
	return instances[instance].ccd->getHyperprior();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_regression_RegressionJNIWrapper_setHyperprior
  (JNIEnv *env, jobject obj, jint instance, jdouble value) {
	instances[instance].ccd->setHyperprior(value);
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_regression_RegressionJNIWrapper_findMode
  (JNIEnv *env, jobject obj, jint instance) {
	fitModel(instances[instance].ccd, *(instances[instance].arguments));
}

extern "C"
JNIEXPORT jint JNICALL Java_dr_inference_regression_RegressionJNIWrapper_getUpdateCount
  (JNIEnv *env, jobject obj, jint instance) {
	return instances[instance].ccd->getUpdateCount();
}

extern "C"
JNIEXPORT jint JNICALL Java_dr_inference_regression_RegressionJNIWrapper_getLikelihoodCount
  (JNIEnv *env, jobject obj, jint instance) {
	return instances[instance].ccd->getLikelihoodCount();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_regression_RegressionJNIWrapper_setPriorType
  (JNIEnv *env, jobject obj, jint instance, jint type) {
 	instances[instance].ccd->setPriorType(type); 
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_regression_RegressionJNIWrapper_makeDirty
  (JNIEnv *env, jobject obj, jint instance) {
  instances[instance].ccd->makeDirty();
}

