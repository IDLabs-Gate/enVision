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

class FaceNet {
    private var tfFacenet : tfWrap?
    
    func load(){
        
        clean()
        tfFacenet = tfWrap()
        tfFacenet?.loadModel("facenet-inf-opt.pb", labels: nil, memMapped: true, optEnv: true)
        tfFacenet?.setInputLayer("input", outputLayer: "embeddings")
    }
    
    func run(image: CIImage)-> [Double]{
        
        guard let tfFacenet = tfFacenet else { return [] }
        
        let inputEdge = 160
        let input = CIImage(cgImage: resizeImage(image, newWidth: CGFloat(inputEdge), newHeight: CGFloat(inputEdge)).cgImage!)
        
        var buffer : CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, inputEdge, inputEdge, kCVPixelFormatType_32BGRA, [String(kCVPixelBufferIOSurfacePropertiesKey) : [:]] as CFDictionary, &buffer)
        
        if let buffer = buffer { CIContext().render(input, to: buffer) }
        
        //neural network forward run
        guard let network_output = tfFacenet.run(onFrame: buffer) else { return [] }
        
        let output = network_output.flatMap{ ($0 as? NSNumber)?.doubleValue }
        
        //print("embeddings", output.count)
        
        return output
    }
    
    static func l2distance(_ feat1: [Double], _ feat2: [Double])-> Double{
        
        //dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
        return sqrt(zip(feat1,feat2).map { f1, f2 in pow(f2-f1,2) }.reduce(0, +))
    }
    
    func clean(){
        tfFacenet?.clean()
        tfFacenet = nil
    }
    
}

typealias FaceOutput = [(face:CIImage, box: CGRect, smile: Bool)]

class FaceDetector {
    
    private let faceDetector = CIDetector(ofType: CIDetectorTypeFace, context: nil, options: [ CIDetectorAccuracy : CIDetectorAccuracyLow ])

    func extractFaces(frame: CIImage)-> FaceOutput {
        
        guard let features = faceDetector?.features(in: frame, options: [CIDetectorSmile : true]) as? [CIFaceFeature] else { return [] }
        
        return features.map { f -> (face: CIImage, box: CGRect, smile: Bool) in
            
            let rect = f.bounds
            let cropped = frame.cropping(to: rect)
            let face = cropped.applying(CGAffineTransform(translationX: -rect.origin.x, y: -rect.origin.y))
            let box = rect.applying(transformToScreen(frame.extent))
            return (face,box, f.hasSmile)
        }
        
    }
    
    func transformToScreen(_ frame: CGRect)-> CGAffineTransform {
        
        let frameWidth = frame.width
        let frameHeight = frame.height
        let screenWidth = UIScreen.main.bounds.width
        let screenHeight = UIScreen.main.bounds.height
        
        //compensate for previewing video frame in AVLayerVideoGravityResizeAspect
        let horizontal = frameWidth/frameHeight > screenWidth/screenHeight
        let seenWidth = horizontal ? screenWidth : screenHeight*frameWidth/frameHeight
        let seenHeight = horizontal ? screenWidth*frameHeight/frameWidth : screenHeight
        let biasX = horizontal ? 0 : (screenWidth-seenWidth)/2
        let biasY = horizontal ? (screenHeight-seenHeight)/2 : 0
        
        let transRatio = frameHeight/seenHeight
        
        //translate CoreImage coordinates to UIKit coordinates
        //also compensate for transRatio of conforming to video frame height
        var transform = CGAffineTransform(scaleX: 1/transRatio, y: -1/transRatio)
        transform = transform.translatedBy(x: (0+biasX)*transRatio, y: (-screenHeight+biasY)*transRatio)
        
        return transform
    }
}
