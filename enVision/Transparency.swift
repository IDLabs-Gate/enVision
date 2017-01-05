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

class Transparency : UIView {

    var boxList = [(rect:CGRect, color:UIColor, label:NSString)]()
    var selector = CGRect.zero
    var drawing = false
    var showSelector = false
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = UIColor.clear
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func draw(_ rect: CGRect) {
        
        guard drawing else { return }
        
        let context = UIGraphicsGetCurrentContext()
        context?.setLineWidth(5)
        
        for b in boxList {
            let textFontAttributes = [
                NSFontAttributeName: UIFont(name: "Helvetica Bold", size: 14)!,
                NSForegroundColorAttributeName: UIColor.white
                ] as [String : Any]
            b.label.draw(at: b.rect.origin+CGPoint(x:9,y:6), withAttributes: textFontAttributes)
            
            context?.setStrokeColor(b.color.withAlphaComponent(0.7).cgColor)
            context?.setFillColor(b.color.withAlphaComponent(0.2).cgColor)
            context?.addRect(b.rect)
            context?.strokePath()
            context?.addRect(b.rect)
            context?.fillPath()
            
        }
        
        if showSelector {
            var selRect = selector
            selRect.origin = selRect.origin - CGPoint(x:5,y:5)
            selRect.size = selRect.size + CGPoint(x:10,y:10)
            context?.setLineWidth(10)
            context?.setStrokeColor(UIColor.white.cgColor)
            context?.addRect(selRect)
            context?.strokePath()
        }
        
    }
    
}

