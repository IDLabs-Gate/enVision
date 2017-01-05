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

#ifndef tfWrap_h
#define tfWrap_h


#endif /* tfWrap_h */

#import <Foundation/Foundation.h>
#import <CoreImage/CoreImage.h>

@interface tfWrap : NSObject
- (void) loadModel:(NSString*)graphFileName labels:(NSString*)labelsFileName;
- (void) loadModel:(NSString*)graphFileName labels:(NSString*)labelsFileName memMapped:(bool)map;
- (void) loadModel:(NSString*)graphFileName labels:(NSString*)labelsFileName memMapped:(bool)map optEnv:(bool)opt;
- (void) setInputLayer:(NSString*)inLayer outputLayer:(NSString*)outLayer;
- (void) setInputMean:(float)mean std:(float)std;
- (NSArray*)runOnFrame:(CVPixelBufferRef)pixelBuffer;
- (NSArray*) getLabels;
- (void) clean;
@end

