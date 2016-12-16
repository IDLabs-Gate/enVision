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

let tfInception = tfWrap()

private var oldPredictions = [String:Double]()
private var labelLayers = [CALayer]()
private let predictionTextLayer = CATextLayer()

extension ViewController {
    
    func loadInceptionModel(){
        
        tfInception.loadModel("InceptionV3-opt.pb", labels: "Imagenet-labels.txt", memMapped: true, optEnv: true)
        tfInception.setInputLayer("Mul", outputLayer: "softmax")
        
        lastModel = tfInception
        
    }
    
    func recognizeInceptionObjects(frameImage:CIImage){
        
        let inputEdge = 299+21
        
        /*let screen_width = UIScreen.main.bounds.width
        let screen_height = UIScreen.main.bounds.height
        let w = frameImage.extent.width
        let h = frameImage.extent.height
        let rect = CGRect(x: (w-h*screen_width/screen_height)/2, y: 0, width: h*screen_width/screen_height, height: h)
        var frameImage = frameImage.cropping(to:rect).applying(CGAffineTransform(translationX: -rect.origin.x, y: -rect.origin.y))
         */
 
        /*let rect = CGRect(x: (frameImage.extent.width-frameImage.extent.height)/2, y: 0, width: frameImage.extent.height, height: frameImage.extent.height)
        var frameImage = frameImage.cropping(to: rect).applying(CGAffineTransform(translationX: -rect.origin.x, y: -rect.origin.y))
        */
        let frameImage = CIImage(cgImage: resizeImage(frameImage, newWidth: CGFloat(inputEdge), newHeight: CGFloat(inputEdge)).cgImage!)
        
        var buffer : CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, Int(frameImage.extent.width), Int(frameImage.extent.height), kCVPixelFormatType_32BGRA, [String(kCVPixelBufferIOSurfacePropertiesKey) : [:]] as CFDictionary, &buffer)
        
        if let buffer = buffer {
            CIContext().render(frameImage, to: buffer)
            
            //let prev = CIImage(cvPixelBuffer: buffer)
            //showPreview(UIImage(ciImage:prev))
        }
        
        let now = Date(timeIntervalSinceNow: 0)
        //neural network forward run
        guard let network_output = tfInception.run(onFrame: buffer) else { return }
        let runTime = Double(Date().timeIntervalSince(now))
        setTimeLabel(time: runTime)
        
        let output = network_output.flatMap{ ($0 as? NSNumber)?.doubleValue }
        let labels = tfInception.getLabels().flatMap{ $0 as? String }
        
        //direct mapping
        let values = output.prefix(labels.count)
        var predictions = [String : Double]()
        
        zip(labels,values).forEach { l, v in
            if v>0.05 {
                predictions[l] = v
            }
        }
        
        async_main {
            self.setPredictionValues(predictions)
            
            self.clearBoxes()//no boxes used with Inception recognition
        }
        
        halt(0.5*runTime)
        
    }
    
    func setPredictionValues(_ newValues: [String:Double]){
        
        print(newValues)
        
        let decayValue = 0.75
        let updateValue = 0.25
        let minimumThreshold = 0.01
        
        var decayedPredictions = [String:Double]()
        
        for p in oldPredictions {
            
            let value = p.value
            
            let decayedPredictionValue = value * decayValue
            
            if decayedPredictionValue > minimumThreshold {
                decayedPredictions[p.key] = decayedPredictionValue
            }
        }
        
        oldPredictions = decayedPredictions
        
        for v in newValues {
            
            let newValue = v.value
            var oldValue = 0.0
            
            if let v = oldPredictions[v.key] {
                oldValue = v
            }
            
            let updatedValue = (oldValue + (newValue * updateValue))
            
            oldPredictions[v.key] = updatedValue
            
        }
        
        var candidateLabels = [(String,Double)]()
        
        for p in oldPredictions {
            if p.value > 0.05 {
                candidateLabels.append(p)
            }
        }
        
        let sortedLabels = candidateLabels.sorted { $0.1 > $1.1 }
        
        let leftMargin = CGFloat(10)
        let topMargin = CGFloat(10)
        
        let valueWidth = CGFloat(48)
        let valueHeight = CGFloat(26)
        
        let labelWidth = CGFloat(246)
        let labelHeight = CGFloat(26)
        
        let labelMarginX = CGFloat(5)
        let labelMarginY = CGFloat(5)
        
        self.removeAllLabelLayers()
        
        var labelCount = 0
        for entry in sortedLabels.prefix(4) {
            
            let label = entry.0
            let value = entry.1
            
            let originY =
                (topMargin + ((labelHeight + labelMarginY) * CGFloat(labelCount)))
            
            let valuePercentage = Int(value * 100.0)
            
            let valueOriginX = leftMargin;
            let valueText = String(format:"%d%%", valuePercentage)
            
            self.addLabelLayerWithText(valueText, originX:valueOriginX, originY:originY, width:valueWidth, height:valueHeight, alignment:kCAAlignmentRight)
            
            let labelOriginX = (leftMargin + valueWidth + labelMarginX)
            
            self.addLabelLayerWithText(label, originX:labelOriginX, originY:originY, width:labelWidth, height:labelHeight, alignment:kCAAlignmentLeft)
            
            if (value > 0.5) {
                //self.speak(label);
            }
            
            labelCount += 1
        }
    }
    
    func removeAllLabelLayers() {
        for layer in labelLayers {
            layer.removeFromSuperlayer()
        }
        labelLayers.removeAll()
    }
    
    func addLabelLayerWithText(_ text:String, originX:CGFloat, originY:CGFloat, width:CGFloat, height:CGFloat, alignment:String) {
        
        let font = "Menlo-Regular"
        let fontSize = CGFloat(18.0)
        
        let marginSizeX = CGFloat(5.0)
        let marginSizeY = CGFloat(2.0)
        
        let backgroundBounds = CGRect(x: originX, y: originY, width: width, height: height)
        
        let textBounds =
            CGRect(x:(originX + marginSizeX), y:(originY + marginSizeY),
                   width: (width - (marginSizeX * 2)), height: (height - (marginSizeY * 2)))
        
        let background = CATextLayer()
        background.backgroundColor = UIColor.black.cgColor
        background.opacity = 0.5
        background.frame = backgroundBounds
        background.cornerRadius = 5.0
        
        self.view.layer.addSublayer(background)
        labelLayers.append(background)
        
        let layer = CATextLayer()
        layer.foregroundColor = UIColor.white.cgColor
        layer.frame = textBounds
        layer.alignmentMode = alignment
        layer.isWrapped = true
        layer.font = font as CFTypeRef
        layer.fontSize = fontSize
        layer.contentsScale = UIScreen.main.scale
        layer.string = text
        self.view.layer.addSublayer(layer)
        labelLayers.append(layer)
    }
    
    func setPredictionText(_ text:String){
        
        predictionTextLayer.foregroundColor = UIColor.white.cgColor
        predictionTextLayer.removeFromSuperlayer()
        view.layer.addSublayer(predictionTextLayer)
        predictionTextLayer.string = text
    }

}
