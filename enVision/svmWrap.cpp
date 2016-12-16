//
//    The MIT License (MIT)
//
//    Copyright (c) 2016 ID Labs L.L.C.
//
//    Permission is hereby granted, free of charge, to any person obtaining a copy
//    of this software and associated documentation files (the "Software"), to deal
//    in the Software without restriction, including without limitation the rights
//    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//    copies of the Software, and to permit persons to whom the Software is
//    furnished to do so, subject to the following conditions:
//
//    The above copyright notice and this permission notice shall be included in all
//    copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//    SOFTWARE.
//

#include "svmWrap.h"
#include "svmutils.h"
#include <stdlib.h>

typedef struct SPredictorInfoStruct {
    struct svm_model* model;
    SLibSvmProblem* problem;
} SPredictorInfo;

void* SVM_create_trainer() {
    SLibSvmTrainingInfo* trainer = create_training_info();
    return trainer;
}

void SVM_destroy_trainer(void* trainerHandle) {
    SLibSvmTrainingInfo* trainer = (SLibSvmTrainingInfo*)(trainerHandle);
    destroy_training_info(trainer);
}

void SVM_train(void* trainerHandle, float expectedLabel, float* predictions, int predictionsLength) {
    SLibSvmTrainingInfo* trainer = (SLibSvmTrainingInfo*)(trainerHandle);
    add_features_to_training_info(trainer, expectedLabel, predictions, predictionsLength);
}

void* SVM_create_predictor_from_trainer(void* trainerHandle) {
    SLibSvmTrainingInfo* trainer = (SLibSvmTrainingInfo*)(trainerHandle);
    SLibSvmProblem* problem = create_svm_problem_from_training_info(trainer);
    const char* parameterCheckError = svm_check_parameter(problem->svmProblem, problem->svmParameters);
    if (parameterCheckError != NULL) {
        fprintf(stderr, "libsvm parameter check error: %s\n", parameterCheckError);
        destroy_svm_problem(problem);
        return NULL;
    }
    struct svm_model* model = svm_train(problem->svmProblem, problem->svmParameters);
    SPredictorInfo* result = (SPredictorInfo*)(malloc(sizeof(SPredictorInfo)));
    result->model = model;
    result->problem = problem;
    return result;
}

void SVM_destroy_predictor(void* predictorHandle) {
    SPredictorInfo* predictorInfo = (SPredictorInfo*)(predictorHandle);
    svm_free_and_destroy_model(&predictorInfo->model);
    if (predictorInfo->problem != NULL) {
        destroy_svm_problem(predictorInfo->problem);
    }
    free(predictorInfo);
}

int SVM_save_predictor(const char* filename, void* predictorHandle) {
    SPredictorInfo* predictorInfo = (SPredictorInfo*)(predictorHandle);
    struct svm_model* model = predictorInfo->model;
    const int saveResult = svm_save_model(filename, model);
    if (saveResult != 0) {
        fprintf(stderr, "Couldn't save libsvm model file to '%s'\n", filename);
        return 0;
    }
    return 1;
}

void* SVM_load_predictor(const char* filename) {
    struct svm_model* model = svm_load_model(filename);
    SPredictorInfo* result = (SPredictorInfo*)(malloc(sizeof(SPredictorInfo)));
    result->model = model;
    result->problem = NULL;
    return result;
}

void SVM_print_predictor(void* predictorHandle) {
    SPredictorInfo* predictorInfo = (SPredictorInfo*)(predictorHandle);
    struct svm_model* model = predictorInfo->model;
    const int saveResult = svm_save_model_to_file_handle(stderr, model);
    if (saveResult != 0) {
        fprintf(stderr, "Couldn't print libsvm model file to stderr\n");
    }
}

float SVM_predict(void* predictorHandle, float* predictions, int predictionsLength) {
    SPredictorInfo* predictorInfo = (SPredictorInfo*)(predictorHandle);
    struct svm_model* model = predictorInfo->model;
    struct svm_node* nodes = create_node_list(predictions, predictionsLength);
    double probabilityEstimates[2];
    svm_predict_probability(model, nodes, probabilityEstimates);
    const double predictionValue = probabilityEstimates[0];
    destroy_node_list(nodes);
    return predictionValue;
}

