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

class ViewController: UIViewController {
    
    override func viewDidLoad() {
        
        super.viewDidLoad()
        
        setupCam()
        
        setupUI()
        
        showSelectionMenu()
        
        /*
        async {
            self.loadFacenetModel()
            
            frameProcessing = { frame in
                self.detectFaces(frameImage: frame)
                oneShot(&finalTask)
            }
        }*/
    }
    
    func setupUI(){
        
        //drawing transparency
        transparency = Transparency(frame: view.bounds)
        view.addSubview(transparency)
        transparency.setActions(tap: tapScreen, press: (pressScreen, duration: 0.3), drag: nil)
        
        //predictions list view
        listView = UIView(frame: CGRect(origin: CGPoint.zero, size: view.bounds.size/3))
        listView.backgroundColor = UIColor.clear
        view.addSubview(listView)
        
        //data slots previews
        setupDataSlots()
        
        //model label
        modelLabel = UILabel(frame: CGRect(x: (UIScreen.main.bounds.width-400)/2, y: 0, width: 400, height: 50))
        modelLabel.textColor = UIColor.white
        modelLabel.textAlignment = .center
        timeLabel.font = .systemFont(ofSize: 18)
        view.addSubview(modelLabel)
        
        //time label
        timeLabel = UILabel(frame: CGRect(x: UIScreen.main.bounds.width-100, y: 0, width: 100, height: 50))
        timeLabel.textColor = UIColor.white
        timeLabel.textAlignment = .center
        timeLabel.font = .systemFont(ofSize: 18)
        view.addSubview(timeLabel)
        
        //threshold labels and steppers
        setupSteppers()
        
    }

}

//MARK: -
enum Model { case yolo; case facenet; case inception; case jetpac; case none }
var currentModel : Model = .none

//MARK: YOLO
let yolo = YOLO()
var yoloThreshold = 0.25
var jyoloThreshold = 60.0
var detectedObjects : YoloOutput = []

extension ViewController {
    
    func loadYoloModel(_ model: Int) {
        
        yolo.load(model)
        currentModel = .yolo
        
        switch model {
        case 1: yoloThreshold = 0.2
        case 2: yoloThreshold = 0.2
        case 3: yoloThreshold = 0.15
        default: yoloThreshold = 0.25
        }
        
        jpac.load()
        jyoloThreshold = 60.0
        
        showSteppers(value1: yoloThreshold, step1: 0.05, value2: jyoloThreshold, step2: 5)
        
        if lastYoloSlots.count>0 {
            dslots = lastYoloSlots
        } else {
            dslots = createDataSlots()
        }
        
        showDataSlots()
        
        showDrawing()
    }
    
    func detectYoloObjects(frameImage:CIImage){
        
        yolo.threshold = yoloThreshold
        detectedObjects = yolo.run(image: frameImage)
        
        drawBoxes(detectedObjects.map { o in
            
            guard dslots.contains(where: {$0.value.first?.label == o.label}) else {
                return (o.box, UIColor.cyan, (String(format: "%.02f ", o.prob)+o.label) as NSString)
            }
            
            let features = jpac.run2(image: o.object)
            
            let slots = dslots.filter{ $0.value.first?.label == o.label }.map{ ($0.value.map { $0.features }, Array(repeating: $0.key.tag, count: $0.value.count)) }
            
            let (pred, dist) = kNN.run(x: features, samples: slots.flatMap{ $0.0 }, classes: slots.flatMap{ $0.1 })
        
            guard pred > -1 else {
                 return (o.box, UIColor.cyan, (String(format: "%.02f ", o.prob)+o.label) as NSString)
            }
            
            print("pred", pred)
            print("min l2 distance", dist)
            
            if dist<jyoloThreshold {
                return (o.box, colorPalette[pred-1], (String(format: "%.02f ", o.prob)+o.label+String(format: " %.02f", dist)) as NSString)
            } else {
                return (o.box, colorPalette[pred-1].gray(factor: 5), (String(format: "%.02f ", o.prob)+o.label+String(format: " %.02f", dist)) as NSString)
            }
        })
        
        print("")
        print(detectedObjects.count, "boxes")
        detectedObjects.forEach { print($0.label, $0.prob) }

    }
    
    func selectObject(tap: CGPoint){
        for o in detectedObjects {
            let object = o.object
            let box = o.box
            
            if box.contains(tap){
                finalTask = {
                    let features = jpac.run2(image: object)
                    
                    let photo = resizeImage(UIImage(ciImage: object), newWidth: 300, newHeight: 300)
                    
                    if var slot = dslots[selectedPrev] {
                        slot.append((features, o.label, photo))
                        dslots[selectedPrev] = slot
                        self.showPreview(selectedPrev, image: UIImage(ciImage:object), number: slot.count)
                    }
                    
                }
                break
            }
        }
    }
    
    func cleanYOLO(){
        yolo.clean()
        jpac.clean()
        hideSteppers()
        lastYoloSlots = dslots
    }
}

//MARK: FaceNet
let fnet = FaceNet()
let kNN = KNN()
var fnetThreshold = 0.75
let fDetector = FaceDetector()

var detectedFaces : FaceOutput = []

extension ViewController {
    
    func loadFacenetModel(){
        
        fnet.load()
        currentModel = .facenet
        
        if lastFaceSlots.count>0 {
            dslots = lastFaceSlots
        } else {
            dslots = createDataSlots()
        }
        
        fnetThreshold = 0.75
        
        showSteppers(value1: fnetThreshold, step1: 0.05)
        
        showDataSlots()
        
        showDrawing()
    }
    
    func detectFaces(frameImage: CIImage){
        
        detectedFaces = fDetector.extractFaces(frame: frameImage)
        
        drawBoxes(detectedFaces.map { f in
            
            guard dslots.contains(where:{ $0.value.count>0 }) else { return (f.box, UIColor.cyan, ("??"+(f.smile ? " 😀 " : "")) as NSString) }
            
            let features = fnet.run(image: f.face)
            
            let slots = dslots.map { ($0.value.map { $0.features }, Array(repeating: $0.key.tag, count: $0.value.count)) }
            
            //let dist = FaceNet.l2distance(features, currentFaceFeatuers)
            
            let (pred, dist) = kNN.run(x: features, samples: slots.flatMap { $0.0 }, classes: slots.flatMap { $0.1 })
            
            guard pred > -1 else { return (f.box, UIColor.cyan, "?!") }
            
            print("pred", pred)
            print("min l2 distance", dist)
            
            let text = String(format: "%.02f", dist)+(f.smile ? " 😀 " : "")
            
            if dist<fnetThreshold {
                return (f.box, colorPalette[pred-1], text as NSString)
            } else {
                return (f.box,  colorPalette[pred-1].gray(factor: 5), text as NSString)
            }
        })
        
    }
    
    func selectFace(tap: CGPoint){
        
        for f in detectedFaces {
            
            let face = f.face
            let box = f.box
            
            if box.contains(tap){
        
                finalTask = {
                    
                    let features = fnet.run(image: face)
                    
                    let photo = resizeImage(UIImage(ciImage: face), newWidth: 200, newHeight: 200)
                    
                    if var slot = dslots[selectedPrev] {
                        slot.append((features, "face", photo))
                        dslots[selectedPrev] = slot
                        self.showPreview(selectedPrev, image: UIImage(ciImage:face), number: slot.count)
                    }
                }
                
                break
            }
        }
    }
    
    func cleanFaceNet(){
        fnet.clean()
        hideSteppers()
        lastFaceSlots =  dslots
    }
    
    func testOtherFaces() {
        
        let e11 = identifyFace(uiImage: #imageLiteral(resourceName: "person1_1"))
        let e12 = identifyFace(uiImage: #imageLiteral(resourceName: "person1_2"))
        
        let e21 = identifyFace(uiImage: #imageLiteral(resourceName: "person2_1"))
        let e22 = identifyFace(uiImage: #imageLiteral(resourceName: "person2_2"))
        
        let e31 = identifyFace(uiImage: #imageLiteral(resourceName: "person3_1"))
        let e32 = identifyFace(uiImage: #imageLiteral(resourceName: "person3_2"))
        
        let (pred, dist) = kNN.run(x: e32, samples: [e11, e12, e21, e22, e31,e32], classes: [1, 1, 2, 2 ,3,3])
        
        print ("class", pred, dist)
        
        /*let dist1 = l2distance(f1, f2)
         let dist2 = l2distance(f3, f4)
         
         let dist3 = l2distance(f1, f3)
         let dist4 = l2distance(f2, f4)
         
         print("same image dist", dist1, dist2)
         print("diff image dist", dist3, dist4)*/
        
    }
    
    func identifyFace(uiImage: UIImage)-> [Double]{
        guard let cgImage = uiImage.cgImage else { return [] }
        guard let f = fDetector.extractFaces(frame: CIImage(cgImage: cgImage)).first else { return [] }
        return fnet.run(image: f.face)
    }
    
}

//MARK: Inception
let incep = Inception()

extension ViewController {
    
    func loadInceptionModel(retrained:Bool=false){
        
        //check for retrained model in bundle
        if retrained{
            if let path1 = Bundle.main.path(forResource: "retrained-opt", ofType: "pb") {
                if let path2 = Bundle.main.path(forResource: "retrained-labels", ofType: "txt"){
                    if FileManager.default.fileExists(atPath: path1) && FileManager.default.fileExists(atPath: path2){
                        
                        incep.loadRetraiend()
                        
                    }
                }
            }
        
        } else {//load default
            
            incep.load()
            
        }

        currentModel = .inception
        
        showPredictionList()
    }
    
    func recognizeInceptionObjects(frameImage:CIImage){
        
        let output = incep.run(image: frameImage)
        
        //UI: prediction list
        var predictions = [String : Double]()
        
        output.forEach { l, p in
            if p>0.05 {
                predictions[l] = p
            }
        }
        
        async_main {
            self.setPredictionValues(predictions)
        }
        
    }
    
    func cleanInception(){
        incep.clean()
    }
    
}

//MARK: Jetpac
let jpac = Jetpac()

extension ViewController {
    
    func loadJetpacModel(){
        
        jpac.load()
        currentModel = .jetpac
        
        showPredictionList()
    }
    
    func recognizeJetpacObjects(frameImage:CIImage){
        
        let output = jpac.run(image: frameImage)
        
        //UI: prob list
        var predictions = [String : Double]()
        output.forEach { l, p in
            if p>0.05 {
                predictions[l] = p
            }
        }
        async_main {
            self.setPredictionValues(predictions)
        }
        
    }
    
    func cleanJetpac(){
        jpac.clean()
    }
}

//MARK: -
var modelLabel = UILabel()
var timeLabel = UILabel()
var finalTask : ((Void)->Void)?

//MARK: Menu UI
var loadLastModel: (Void)->Void = {}

extension ViewController {
    
    func showSelectionMenu() {
        
        let alert1 = UIAlertController(title: "Select Model", message: "", preferredStyle: .alert)
        
        alert1.addAction(UIAlertAction(title: "YOLO", style: .default, handler: { (_) in
            
            let alert2 = UIAlertController(title: "Select YOLO Model", message: "", preferredStyle: .alert)
            
            alert2.addAction(self.createMenuAction(title: "YOLO 1 tiny (VOC)", loadTask: { () in
                self.loadYoloModel(1)
            }, frameTask: { (frame) in
                self.detectYoloObjects(frameImage: frame)
            }))
            
            alert2.addAction(self.createMenuAction(title: "YOLO 1 small (VOC)" , loadTask: { () in
                self.loadYoloModel(2)
            }, frameTask: { (frame) in
                self.detectYoloObjects(frameImage: frame)
            }))
            
            alert2.addAction(self.createMenuAction(title: "YOLO 1.1 tiny (COCO)", loadTask: { () in
                self.loadYoloModel(3)
            }, frameTask: { (frame) in
                self.detectYoloObjects(frameImage: frame)
            }))
            
            alert2.addAction(self.createMenuAction(title: "YOLO 2 (COCO)", loadTask: { () in
                self.loadYoloModel(0)
            }, frameTask: { (frame) in
                self.detectYoloObjects(frameImage: frame)
            }))
            
            alert2.addAction(UIAlertAction(title: "Back", style: .destructive, handler: { (_) in
                async_main {
                    self.present(alert1, animated: true) {}
                }
            }))
            
            async_main {
                self.present(alert2, animated: true) {}
            }
            
        }))
        
        alert1.addAction(createMenuAction(title: "FaceNet (FaceScrub & CASIA-Webface)", shortTitle: true, loadTask: { () in
            self.loadFacenetModel()
        }, frameTask: { (frame) in
            self.detectFaces(frameImage: frame)
        }))
        
        alert1.addAction(createMenuAction(title: "Inception v3 (ImageNet)", shortTitle: true, loadTask: { () in
            self.loadInceptionModel()
        }, frameTask: { (frame) in
            self.recognizeInceptionObjects(frameImage: frame)
        }))
        
        alert1.addAction(createMenuAction(title: "Re-Inception (Custom Data)", shortTitle: true, loadTask: { () in
            self.loadInceptionModel(retrained: true)
        }, frameTask: { (frame) in
            self.recognizeInceptionObjects(frameImage: frame)
        }))
        
        alert1.addAction(createMenuAction(title: "Jetpac (ImageNet)", shortTitle: true, loadTask: { () in
            self.loadJetpacModel()
        }, frameTask: { (frame) in
            self.recognizeJetpacObjects(frameImage: frame)
        }))
        
        alert1.addAction(UIAlertAction(title: "Cancel", style: .destructive, handler: { (_) in
            loadLastModel()
        }))
        
        async_main {
            self.present(alert1, animated: true) {}
        }
        
    }
    
    func createMenuAction(title: String, shortTitle: Bool = false, style:UIAlertActionStyle = .default, loadTask: @escaping (Void)->Void, frameTask:@escaping (CIImage)->Void)-> UIAlertAction {
        
        return UIAlertAction(title: shortTitle ? title.components(separatedBy: " ").first : title, style: style) { _ in
            loadLastModel = { self.view.addActivityIndicatorOverlay(){ remove in
                
                loadTask()
                modelLabel.text = title
                
                frameProcessing = { frame in
                    let now = Date(timeIntervalSinceNow: 0)
                    
                    frameTask(frame)
                    oneShot(&finalTask)
                    
                    let runTime = Double(Date().timeIntervalSince(now))
                    self.setTimeLabel(time: runTime)
                    
                }
                remove()
                
                }
            }
            loadLastModel()
            
        }
    }
    
    func setTimeLabel(time: Double){
        async_main { timeLabel.text = time<60 ? String(format: "%.2f",time) : "--" }
    }
    
    func cleanAll(){
        frameProcessing = nil
        
        hidePredictionList()
        hideDrawing()
        hideDataSlots()
        
        switch currentModel {
        case .yolo: cleanYOLO()
        case .facenet: cleanFaceNet()
        case .inception: cleanInception()
        case .jetpac: cleanJetpac()
        default: break
        }
        
    }
}

//MARK: Gestures
extension ViewController {
    
    func tapScreen(view: UIView) {
        guard let gesture = view.tapGesture else { return }
        let place = gesture.location(in: view)
        
        switch currentModel {
        case .facenet: selectFace(tap: place)
        case .yolo: selectObject(tap: place)
        default: break
        }
        
        
    }
    
    func pressScreen(view: UIView){
        guard let gesture = view.pressGesture else { return }
        guard gesture.state == .began else { return }
        
        cleanAll()
        showSelectionMenu()

    }
    
    func tapPreview(view: UIView) {
        guard let prev = view as? UIImageView else { return }
        
        selectedPrev.backgroundColor = UIColor.gray.withAlphaComponent(0.3)
        selectedPrev = prev
        selectedPrev.backgroundColor = UIColor.white.withAlphaComponent(0.5)
        
    }
    
    func tap2Preview(view: UIView) {
        guard let prev = view as? UIImageView else { return }
        
        decDataSlot(view: prev)
    }
    
    func pressPreview(view: UIView){
        guard let prev = view as? UIImageView else { return }
        
        clearDataSlot(view: prev)
    }
}

//MARK: Data Slots UI

typealias DataSlots = [UIImageView : [(features: [Double], label: String, photo: UIImage)]]
var dslots = DataSlots()
var lastFaceSlots = DataSlots()
var lastYoloSlots = DataSlots()
var previews = [UIImageView]()

var selectedPrev = UIImageView() {
    didSet {
        transparency.selector = selectedPrev.frame
        transparency.setNeedsDisplay()
    }
}

extension ViewController {
    
    func setupDataSlots(){
        let n = 5
        let prevEdge = CGFloat(160)
        let space = CGFloat(20)
        
        for i in 0..<n {
            
            let shift = (UIScreen.main.bounds.width-(CGFloat(n)*(prevEdge+space)))/2
            let preview = UIImageView(frame:CGRect(x: shift+CGFloat(i)*(prevEdge+space), y: UIScreen.main.bounds.height-prevEdge, width: prevEdge, height: prevEdge) )
            preview.backgroundColor = UIColor.gray.withAlphaComponent(0.3)
            preview.isHidden = true
            preview.isUserInteractionEnabled = true
            view.addSubview(preview)
            preview.setActions(tap: tapPreview, press: (pressPreview, duration: 0.5), drag: nil)
            preview.setDoubleTouchTapAction(tap2Preview)
            preview.tag = i+1
            
            if i==0 {
                selectedPrev = preview
                selectedPrev.backgroundColor = UIColor.white.withAlphaComponent(0.5)
            }
            
            previews.append(preview)
        }
    }
    
    func createDataSlots()->DataSlots{
        var slots = DataSlots()
        previews.forEach { slots[$0] = [] }
        return slots
    }
    func decDataSlot(view: UIImageView){
        
        guard var slot = dslots[view] else { return }
        
        if slot.count>0 {
            slot.removeLast()
            dslots[view] = slot
            showPreview(view, image: slot.last?.photo, number: slot.count)
        }
    }
    
    func clearDataSlot(view: UIImageView) {
        dslots[view] = []
        self.hidePreview(view)
    }
    
    func hideDataSlots(){
        
        async_main {
            dslots.forEach {
                let prev = $0.key
                prev.image = nil
                prev.subviews.forEach { $0.removeFromSuperview() }
                prev.isHidden = true
            }
        }
        
        hideSelector()
    }
    
    func showDataSlots(){
        
        async_main {
            dslots.forEach {
                let prev = $0.key
                prev.isHidden = false
                
                let slot = $0.value
                self.showPreview(prev, image: slot.last?.photo, number: slot.count)
            }
        }
        
        showSelector()
    }
    
    func showPreview(_ view: UIImageView, image: UIImage?, number: Int = -1){
        
        guard let image = image else { hidePreview(view); return }
        async_main {
            
            view.image = image
            
            view.subviews.forEach{ $0.removeFromSuperview() }
            let l = UILabel(); l.textColor = UIColor.black
            l.backgroundColor = colorPalette[view.tag-1]
            l.font = UIFont(name: "Helvetica Bold", size: 14)!
            l.textAlignment = .center
            l.frame.size = CGSize(width: 30, height: 25)
            l.text = number>0 ? String(number) : ""
            view.addSubview(l)
        }
    }
    
    func hidePreview(_ view: UIImageView){
        async_main { view.image = nil; view.subviews.forEach { $0.removeFromSuperview() } }
    }
   
}

//MARK: Drawing Transparency
var transparency = Transparency()
var colorPalette = [UIColor.red, UIColor.blue, UIColor.green, UIColor.yellow, UIColor.magenta]

extension ViewController {
    
    func drawBoxes(_ boxes: [(CGRect, UIColor, NSString)]){
        
        async_main {
            transparency.boxList.removeAll()
            transparency.boxList = boxes
            transparency.setNeedsDisplay()
        }
    }
    
    func clearBoxes(){
        async_main {
            self.drawBoxes([])
            transparency.setNeedsDisplay()
        }
    }
    
    func showSelector(){
        async_main {
            transparency.showSelector = true
            transparency.setNeedsDisplay()
        }
    }
    
    func hideSelector(){
        async_main {
            transparency.showSelector = false
            transparency.setNeedsDisplay()
        }
        
    }
    
    func showDrawing(){
        async_main {
            transparency.drawing = true
            transparency.setNeedsDisplay()
        }
    }
    
    func hideDrawing(){
        
        clearBoxes()
        
        async_main {
            transparency.drawing = false
            transparency.setNeedsDisplay()
        }
    }

}

//MARK: Threshold Steppers
var stepper = UIStepper()
var stepper2 = UIStepper()
var thresholdLabel = UILabel()
var thresholdLabel2 = UILabel()

extension ViewController {
    
    func setupSteppers(){
        thresholdLabel.bounds = stepper.bounds
        thresholdLabel.frame.origin = CGPoint(x: 10,y: 10)
        thresholdLabel.textColor = UIColor.white
        thresholdLabel.textAlignment = .center
        thresholdLabel.font = .systemFont(ofSize: 18)
        view.addSubview(thresholdLabel)
        
        thresholdLabel2.bounds = stepper2.bounds
        thresholdLabel2.frame.origin = CGPoint(x: 50+stepper2.bounds.width,y: 10)
        thresholdLabel2.textColor = UIColor.white
        thresholdLabel2.textAlignment = .center
        thresholdLabel2.font = .systemFont(ofSize: 18)
        view.addSubview(thresholdLabel2)
        
        stepper.frame.origin = CGPoint(x: 10, y: 20+stepper.bounds.height)
        stepper.tintColor = UIColor.cyan
        stepper.addTarget(self, action: #selector(ViewController.stepper1Action), for: .touchUpInside)
        view.addSubview(stepper)
        stepper2.frame.origin = CGPoint(x: 50+stepper2.bounds.width, y: 20+stepper2.bounds.height)
        stepper2.tintColor = UIColor.cyan
        stepper2.addTarget(self, action: #selector(ViewController.stepper2Action), for: .touchUpInside)
        view.addSubview(stepper2)
        
        hideSteppers()
    }
    
    func showSteppers(value1: Double, step1: Double, value2: Double = -1, step2: Double=1){
        
        stepper.isHidden = false
        stepper.value = value1
        stepper.stepValue = step1
        stepper.maximumValue = step1*25
        
        thresholdLabel.isHidden = false
        thresholdLabel.text = String(value1)
        
        if value2>=0 {
            stepper2.isHidden = false
            stepper2.value = value2
            stepper2.stepValue = step2
            stepper2.maximumValue = step2*25
            
            thresholdLabel2.isHidden = false
            thresholdLabel2.text = String(value2)
        }
    }
    
    func hideSteppers(){
        stepper.isHidden = true
        stepper2.isHidden = true
        thresholdLabel.isHidden = true
        thresholdLabel2.isHidden = true
        
        thresholdLabel.text = ""
        thresholdLabel.text = ""
        
    }
    
    func stepper1Action(sender: UIStepper){
        
        yoloThreshold = sender.value
        fnetThreshold = sender.value
        
        thresholdLabel.text = String(format:"%.02f", sender.value)
        
    }
    
    func stepper2Action(sender: UIStepper){
        
        jyoloThreshold = sender.value
        
        thresholdLabel2.text = String(format:"%.02f", sender.value)
    }
}

//MARK: Prediction list UI
var listView = UIView()
var oldPredictions = [String:Double]()
var labelLayers = [CALayer]()
let predictionTextLayer = CATextLayer()

extension ViewController {
    
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
        
        listView.layer.addSublayer(background)
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
        listView.layer.addSublayer(layer)
        labelLayers.append(layer)
    }
    
    func setPredictionText(_ text:String){
        
        predictionTextLayer.foregroundColor = UIColor.white.cgColor
        predictionTextLayer.removeFromSuperlayer()
        view.layer.addSublayer(predictionTextLayer)
        predictionTextLayer.string = text
    }
    
    func hidePredictionList(){
        listView.isHidden = true
    }
    
    func showPredictionList(){
        listView.isHidden = false
    }
    
    /*func speak(_ words: String){
     guard !synth.isSpeaking else { return }
     let utterance = AVSpeechUtterance(string: words)
     utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
     utterance.rate = 0.75*AVSpeechUtteranceDefaultSpeechRate
     synth.speak(utterance)
     }*/
    
}

