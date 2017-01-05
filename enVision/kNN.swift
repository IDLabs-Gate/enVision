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

class KNN {

    private let kNN = tfkNN()
    private var loaded = false

    func load(){
        kNN.loadModel("kNN.pb")
        loaded = true
    }
    
    func run(x: [Double], samples: [[Double]], classes: [Int]) -> (Int, Double){
        
        guard classes.count>0 && samples.count==classes.count else { return (-1,0) }
        
        if !loaded { load() }
        
        //choosing k = n^(1/2), but not more than the min count/2 of samples in a class
        let minCount = Set(classes).map { c in classes.filter { $0==c }.count }.min{ a,b in a<b }!
        var k = Int(sqrt(Double(samples.count)))
        if k>minCount/2 { k = minCount/2 }
        if k<1 { k = 1 }
        print ("K", k)
        
        let c = kNN.classify(x, samples: samples, classes: classes, k: Int32(k))
        
        guard let pred = c?.first?.key as? Int else { return (-1,0) }
        guard let dist = c?.first?.value as? Double else { return (-1,0) }
        
        return (pred,dist)
    }
    
    func clean(){
        kNN.clean()
        loaded = false;
    }
}
