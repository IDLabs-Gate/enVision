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

let tfFacenet = tfWrap()

private let faceDetector = CIDetector(ofType: CIDetectorTypeFace, context: nil, options: [ CIDetectorAccuracy : CIDetectorAccuracyLow ])

private var frameRects : [CGRect] = []
private var frame : CIImage? = nil

private var otherFaceFeatures : [[Double]] = []
private var currentFaceFeatuers : [Double] = []

extension ViewController {
    
    func loadFacenetModel(){
        
            tfFacenet.loadModel("facenet-inf-opt.pb", labels: nil, memMapped: true, optEnv: true)
            tfFacenet.setInputLayer("input", outputLayer: "embeddings")
            
            lastModel = tfFacenet
        
    }
    
    func selectFace(tap: CGPoint) {
       
        guard let frameImage = frame else { return }
        
        var tapInside = false
        
        for rect in frameRects {
            
            if rect.applying(transformToScreen(frameImage.extent)).contains(tap){
                tapInside = true
                let cropped = frameImage.cropping(to: rect)
                let face = cropped.applying(CGAffineTransform(translationX: -rect.origin.x, y: -rect.origin.y))
                
                finalTask = {
                    currentFaceFeatuers = self.generateFaceFeatures(faceImage: face)
                    self.showPreview(UIImage(ciImage:face), edge: 160)
                }
                
                break
            }
        }
        
        if !tapInside {
            finalTask = {
                currentFaceFeatuers.removeAll()
                self.hidePreview()
            }
        }
    }
    
    func detectFaces(frameImage: CIImage){
        
        guard let features = faceDetector?.features(in: frameImage) as? [CIFaceFeature] else { return }
        
        frame = frameImage
        frameRects = features.map { $0.bounds }
        
        var boxesToDraw : [CGRect] = []
        
        for rect in frameRects {
            //let cropped = frameImage.cropping(to: rect)
            //let face = cropped.applying(CGAffineTransform(translationX: -rect.origin.x, y: -rect.origin.y))
            
            //let features = self.generateFaceFeatures(faceImage: face)
            
            let dist = 0.0 // l2distance(features, currentFaceFeatuers)
            //print("face l2 distance", dist)
            
            if dist<1.0 {
                boxesToDraw.append(rect.applying(transformToScreen(frameImage.extent)))
            }
        }
        
        drawBoxes(boxesToDraw, color: true)
        
    }
    
    func generateFaceFeatures(faceImage: CIImage)->[Double]{
        
        let inputEdge = 160
        let input = CIImage(cgImage: resizeImage(faceImage, newWidth: CGFloat(inputEdge), newHeight: CGFloat(inputEdge)).cgImage!)
        
        var buffer : CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, inputEdge, inputEdge, kCVPixelFormatType_32BGRA, [String(kCVPixelBufferIOSurfacePropertiesKey) : [:]] as CFDictionary, &buffer)
        
        if let buffer = buffer { CIContext().render(input, to: buffer) }
        
        //neural network forward run
        guard let network_output = tfFacenet.run(onFrame: buffer) else { return [] }
        
        let output = network_output.flatMap{ ($0 as? NSNumber)?.doubleValue }
        
        print("embeddings", output.count)
        
        return output
    }
    
    func l2distance(_ feat1: [Double], _ feat2: [Double])-> Double{
        
        //dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
        return sqrt(zip(feat1,feat2).map { f1, f2 in pow(f2-f1,2) }.reduce(0, +))
        
    }
    
    
    func testOtherFaces() {
        
        let f1 = identifyFace(uiImage: #imageLiteral(resourceName: "person1_1.jpg"))
        let f2 = identifyFace(uiImage: #imageLiteral(resourceName: "person1_2.jpg"))
        
        let f3 = identifyFace(uiImage: #imageLiteral(resourceName: "person2_1.jpg"))
        let f4 = identifyFace(uiImage: #imageLiteral(resourceName: "person2_2.jpg"))
        
        let dist1 = l2distance(f1, f2)
        let dist2 = l2distance(f3, f4)
        
        let dist3 = l2distance(f1, f3)
        let dist4 = l2distance(f2, f4)
        
        print("same image dist", dist1, dist2)
        print("diff image dist", dist3, dist4)
        
    }
    
    func identifyFace(uiImage: UIImage)-> [Double]{
        guard let cgImage = uiImage.cgImage else { return [] }
        return identifyFace(ciImage: CIImage(cgImage: cgImage))
        
    }
    
    func identifyFace(ciImage: CIImage)-> [Double]{
        
        guard let faceFeat = faceDetector?.features(in: ciImage).first as? CIFaceFeature else { return [] }
        let rect = faceFeat.bounds
        let cropped = ciImage.cropping(to: rect)
        let face = cropped.applying(CGAffineTransform(translationX: -rect.origin.x, y: -rect.origin.y))
        
        return self.generateFaceFeatures(faceImage: face)
        
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
