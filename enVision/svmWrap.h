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


#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void* SVM_create_trainer();
void SVM_destroy_trainer(void* trainerHandle);
void SVM_train(void* trainerHandle, float expectedLabel, float* predictions, int predictionsLength);
void* SVM_create_predictor_from_trainer(void* trainerHandle);
void SVM_destroy_predictor(void* predictorHandle);
int SVM_save_predictor(const char* filename, void* predictorHandle);
void* SVM_load_predictor(const char* filename);
void SVM_print_predictor(void* predictorHandle);
float SVM_predict(void* predictorHandle, float* predictions, int predictionsLength);

#ifdef __cplusplus
}
#endif // __cplusplus
