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

import Foundation

extension ViewController {
    
    func createSVMPredictor(features: [Float], otherFeatures: [[Float]])->UnsafeMutableRawPointer?{
        guard otherFeatures.count>0 else { return nil }
        
        let now = Date(timeIntervalSinceNow: 0)
        
        let trainer = SVM_create_trainer()
        var i = 0
        print("features dim", features.count)
        print("other dim", otherFeatures.first!.count)
        for other in otherFeatures {
            SVM_train(trainer, 1.0, UnsafeMutablePointer<Float>(mutating: features), Int32(features.count))
            SVM_train(trainer, 0.0, UnsafeMutablePointer<Float>(mutating: other), Int32(other.count))
            i+=1
        }
        
        let predictor = SVM_create_predictor_from_trainer(trainer)
        SVM_destroy_trainer(trainer)
        
        print("Predictor time = "+String(Double(Date().timeIntervalSince(now)))+" for iterations: "+String(i))
        
        return predictor
    }
    
    //let prob = SVM_predict(predictor, UnsafeMutablePointer<Float>(mutating: features), Int32(features.count))
    

}
