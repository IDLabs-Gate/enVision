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

import UIKit

class Inception {
    
    private var tfInception : tfWrap?
    
    func load() {
        
        clean()
        tfInception = tfWrap()
        tfInception?.loadModel("InceptionV3-opt.pb", labels: "Imagenet-labels.txt", memMapped: true, optEnv: true)
        tfInception?.setInputLayer("Mul", outputLayer: "softmax")
    }
    
    func loadRetraiend(){
        clean()
        tfInception = tfWrap()
        tfInception?.loadModel("retrained-opt.pb", labels: "retrained-labels.txt", memMapped: true, optEnv: true)
        tfInception?.setInputLayer("Mul", outputLayer: "final_result")
    }
    
    func run(image: CIImage) -> [(label: String, prob: Double)] {
        
        guard let tfInception = tfInception else { return [] }
        
        let inputEdge = 299+21
        /*let screen_width = UIScreen.main.bounds.width
         let screen_height = UIScreen.main.bounds.height
         let w = frameImage.extent.width
         let h = frameImage.extent.height
         let rect = CGRect(x: (w-h*screen_width/screen_height)/2, y: 0, width: h*screen_width/screen_height, height: h)
         var frameImage = frameImage.cropping(to:rect).applying(CGAffineTransform(translationX: -rect.origin.x, y: -rect.origin.y))
         */
        
        let frameImage = CIImage(cgImage: resizeImage(image, newWidth: CGFloat(inputEdge), newHeight: CGFloat(inputEdge)).cgImage!)
        
        var buffer : CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, Int(frameImage.extent.width), Int(frameImage.extent.height), kCVPixelFormatType_32BGRA, [String(kCVPixelBufferIOSurfacePropertiesKey) : [:]] as CFDictionary, &buffer)
        
        if let buffer = buffer {
            CIContext().render(frameImage, to: buffer)
        }
        
        //neural network forward run
        guard let network_output = tfInception.run(onFrame: buffer) else { return [] }
        
        let output = network_output.flatMap{ ($0 as? NSNumber)?.doubleValue }
        let labels = tfInception.getLabels().flatMap{ $0 as? String }
        
        return Array(zip(labels, output.prefix(labels.count)))
    }
    
    func clean() {
        tfInception?.clean()
        tfInception = nil
    }
}
