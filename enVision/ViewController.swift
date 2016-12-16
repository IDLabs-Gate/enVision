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

var frameProcessing: ((CIImage)->Void)?

var finalTask : ((Void)->Void)?

private let alert = UIAlertController(title: "Select Model", message: "", preferredStyle: .alert)

private var loadLastModel: (Void)->Void = {}
var lastModel = tfWrap()

class ViewController: UIViewController {
    
    var worldView: WorldView!
    var preview: UIImageView!
    var timeLabel: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        worldView = WorldView(frame: view.bounds)
        worldView.backgroundColor = UIColor.black
        view.addSubview(worldView)
        worldView.setActions(tap: tapScreen, press: nil, drag: nil)

        preview = UIImageView()
        view.addSubview(preview)

        timeLabel = UILabel(frame: CGRect(x: UIScreen.main.bounds.width-100, y: 0, width: 100, height: 50))
        timeLabel.textColor = UIColor.white
        timeLabel.textAlignment = .center
        timeLabel.font = .systemFont(ofSize: 18)
        view.addSubview(timeLabel)
        
        setupCam()
        
        createSelectionMenu()
        
        async_main {
            self.present(alert, animated: true) {}
        }
        
    }

    func createSelectionMenu() {
        
        alert.addAction(createMenuAction(title: "Inception (ImageNet)", loadTask: { () in
            self.loadInceptionModel()
        }, frameTask: { (frame) in
            self.recognizeInceptionObjects(frameImage: frame)
        }))
        
        let yolo = singleton_handle()
        //let face = singleton_handle()
        
        alert.addAction(createMenuAction(title: "YOLO 1 tiny (VOC)", loadTask: { () in
            self.loadYoloModel(tiny: true)
        }, frameTask: { (frame) in
            async_singleton(yolo){
                self.detectYoloObjects(frameImage: frame)
            }
            //async_singleton(face){
            //    self.detectFaces(frameImage: frame)
            //}
        }))
        
        alert.addAction(createMenuAction(title: "YOLO 1 small (VOC)" , loadTask: { () in
            self.loadYoloModel()
        }, frameTask: { (frame) in
            self.detectYoloObjects(frameImage: frame)
        }))
        
        alert.addAction(createMenuAction(title: "YOLO 1.1 tiny (COCO)", loadTask: { () in
            self.loadYoloModel(tiny: true, coco: true)
        }, frameTask: { (frame) in
            self.detectYoloObjects(frameImage: frame)
        }))
        
        alert.addAction(createMenuAction(title: "YOLO 2 (COCO)", loadTask: { () in
            self.loadYoloModel(coco: true, v2: true)
        }, frameTask: { (frame) in
            self.detectYoloObjects(frameImage: frame)
        }))
        
        /*alert.addAction(createMenuAction(title: "FaceNet", loadTask: { () in
            self.loadFacenetModel()
            self.testOtherFaces()
        }, frameTask: { (frame) in
            self.detectFaces(frameImage: frame)
        }))*/
        
        alert.addAction(UIAlertAction(title: "Cancel", style: .destructive, handler: { (_) in
            loadLastModel()
        }))
        
    }
    
    func createMenuAction(title: String, style:UIAlertActionStyle = .default, loadTask: @escaping (Void)->Void, frameTask:@escaping (CIImage)->Void)-> UIAlertAction {
        
        return UIAlertAction(title: title, style: style) { _ in
            loadLastModel = { self.view.addActivityIndicatorOverlay(){ remove in
                
                loadTask()
                frameProcessing = { frame in
                    frameTask(frame)
                    oneShot(&finalTask)
                }
                remove()
                
                }
            }
            loadLastModel()
            
        }
    }
    
    func tapScreen() {
        
        //guard let gesture = worldView.tapGesture else { return }
        //let place = gesture.location(in: worldView)
        //selectFace(tap: place)
        
        frameProcessing = nil
        lastModel.clean()
        
        self.hidePreview()
        self.present(alert, animated: true) {}
    }
    
    func showPreview(_ image: UIImage, edge: CGFloat){
        
        async_main {
            self.preview.frame = CGRect(x: 0.5*UIScreen.main.bounds.width-edge/2, y: UIScreen.main.bounds.height-edge, width: edge, height: edge)
            self.preview.image = image
        }
    }
    
    func showPreview(_ image: UIImage){
        
        async_main {
            self.preview.frame = CGRect(x: 0.5*UIScreen.main.bounds.width-image.size.width/2, y: UIScreen.main.bounds.height-image.size.height, width: image.size.width, height: image.size.height)
            self.preview.image = image
        }
    }
    
    func hidePreview(){
        async_main { self.preview.image = nil }
    }
    
    func drawBoxes(_ boxes: [CGRect]){
        drawBoxes(boxes, color: false)
    }
    
    func drawBoxes(_ boxes: [CGRect], color: Bool){
        
        async_main {
            
            let layers = self.view.layer.sublayers
            
            CATransaction.begin()
            
            CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
            
            if let subLayers = layers {
                
                for l in subLayers {
                    if l.name == (color ? "BoxLayer_colored" : "BoxLayer") {
                        l.isHidden = true
                    }
                }
                
            }
            
            guard boxes.count > 0 else { CATransaction.commit(); return }
            
            var currentSublayer = 0
            
            for boxRect in boxes {
                
                var boxLayer: CALayer? = nil
                
                //re-use existing layer if possible
                if let subLayers = layers {
                    
                    while boxLayer==nil && currentSublayer<subLayers.count {
                        let currentLayer = subLayers[currentSublayer]
                        currentSublayer+=1
                        if currentLayer.name == (color ? "BoxLayer_colored" : "BoxLayer") {
                            boxLayer = currentLayer
                        }
                    }
                }
                
                if let layer = boxLayer {
                    layer.frame = boxRect
                    layer.isHidden = false
                }
                    
                else {
                    //create new one if necessary
                    let newBoxLayer = CALayer()
                    newBoxLayer.contents = color ? #imageLiteral(resourceName: "square2").cgImage : #imageLiteral(resourceName: "square").cgImage
                    newBoxLayer.name = color ? "BoxLayer_colored" : "BoxLayer"
                    newBoxLayer.frame = boxRect
                    self.view.layer .addSublayer(newBoxLayer)
                    
                }
            }
            
            CATransaction.commit()
        }
    }
        
    func clearBoxes(){
        
        drawBoxes([], color: true)
        drawBoxes([], color: false)
    }
    
    func setTimeLabel(time: Double){
        async_main { self.timeLabel.text = String(format: "%.2f",time)}
    }
    /*func speak(_ words: String){
        guard !synth.isSpeaking else { return }
        let utterance = AVSpeechUtterance(string: words)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.75*AVSpeechUtteranceDefaultSpeechRate
        synth.speak(utterance)
    }*/
    
    
    //MARK: - Orientation
    
    override var shouldAutorotate : Bool {
        
        return true
    }
    
    override var supportedInterfaceOrientations : UIInterfaceOrientationMask {
        return .landscape
    }
    
    override func viewWillTransition(to size: CGSize, with coordinator: UIViewControllerTransitionCoordinator) {
        
        orientCam()
    }
    
    override var prefersStatusBarHidden : Bool {
        return true
    }
    
}

