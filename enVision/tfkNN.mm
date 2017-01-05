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

#import "tfkNN.h"
#import "tensorflow_utils.h"

@interface tfkNN() {
    
    std::unique_ptr<tensorflow::Session> tf_session;
    dispatch_queue_t kNNQueue;
}

@end

@implementation tfkNN

-(void) loadModel:(NSString*)modelName{
    
    [self createQueue];
    dispatch_sync(kNNQueue,^{
        NSString* model_file_name;
        NSString* model_file_type;
        
        NSArray* strArr = [modelName componentsSeparatedByString:@"."];
        
        if (strArr.count>0) {
            model_file_name = strArr[0];
            
            if (strArr.count>1) {
                model_file_type = strArr[1];
            } else {
                model_file_type = @"pb";
            }
            tensorflow::Status load_status;
            
            load_status = LoadModel(model_file_name, model_file_type, &tf_session);
            
            if (!load_status.ok()) {
                LOG(FATAL) << "Couldn't load model: " << load_status;
            }
            
        }
    });
}

-(NSDictionary*) classify:(NSArray*)x samples:(NSArray*)samples classes:(NSArray*)classes k:(int)k{
    
    __block NSDictionary *output = [NSDictionary dictionary];
    
    //class count must be same as sample count
    //auto classCount = [NSSet setWithArray:classes].count;
    
    //K must be less than or equal sampleCount
    if (k>samples.count){ k = (int)samples.count; }
    //LOG(INFO)<<"K = "<<k;
    
    [self createQueue];
    dispatch_sync(kNNQueue, ^{
        
        //Xte
        auto length = static_cast<long long>(x.count);
        tensorflow::Tensor test(tensorflow::DT_FLOAT, tensorflow::TensorShape({length}));
        
        float *testData = test.tensor<float,1>().data();
        for(int i=0; i<length; ++i){
            testData[i] = [x[i] floatValue];
        }
        
        //Xtr - training samples
        auto sampleCount = static_cast<long long>(samples.count);
        tensorflow::Tensor train(tensorflow::DT_FLOAT, tensorflow::TensorShape({sampleCount, length}));
        float *trainData = train.tensor<float,2>().data();
        
        for(int s=0; s<sampleCount; ++s){
            float*trainData_item = trainData+(s*length);
            NSArray* sample = samples[s];
            for(int i=0; i<length; ++i){
                trainData_item[i] = [sample[i] floatValue];
            }
        }
        
        //Ytr - training samples classes
        tensorflow::Tensor train_classes(tensorflow::DT_INT64, tensorflow::TensorShape({sampleCount}));
        
        int64_t *classData = train_classes.tensor<int64_t, 1>().data();
        
        for(int c=0; c<classes.count; ++c){
            classData[c] = [classes[c] intValue];
        }
        
        //K - number of nearest neighbors to consider
        tensorflow::Tensor K(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        
        int64_t *Kdata = K.tensor<int64_t,0>().data();
        Kdata[0] = k;
        
        if (tf_session.get()){
            
            std::vector<tensorflow::Tensor> outputs;
            tensorflow::Status run_status = tf_session->Run
            ({{"Xte", test}, {"Xtr", train}, {"Ytr", train_classes}, {"K", K}},
             {"pred", "dist"},
             {}, &outputs);
            
            if (!run_status.ok()) {
                LOG(ERROR) << "Running model failed:" << run_status;
            } else {
                //LOG(INFO)<< "Run duration: "<< -[startTime timeIntervalSinceNow];
                
                tensorflow::Tensor *predTensor = &outputs[0];
                tensorflow::Tensor *distTensor = &outputs[1];

                auto predValue = predTensor->flat<int>();
                auto distValue = distTensor->flat<float>();
                
                output = @{@(predValue(0)): @(distValue(0))};
                
            }
        }
        
    });
    
    return output;
}


-(void) clean {
    [self createQueue];
    dispatch_sync(kNNQueue, ^{
        CleanSession(&tf_session);
    });
}

-(void) createQueue {
    if (kNNQueue==nil) {
        kNNQueue = dispatch_queue_create("kNNQueue", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_DEFAULT, 0) );
    }
}


@end


//python code to buid kNN.pb

/*
 import tensorflow as tf
 
 # K = 2
 
 # tf Graph Input
 K = tf.placeholder("int64", None, name='K')
 xtr = tf.placeholder("float", [None, None], name='Xtr') # [None, 1792]
 ytr = tf.placeholder("int64", [None], name = 'Ytr') # [None, 2]
 xte = tf.placeholder("float", [None], name = 'Xte') # [None, 1792]
 
 # Euclidean Distance
 distance = tf.neg(tf.sqrt(tf.reduce_sum(tf.square(tf.sub(xtr, xte)), reduction_indices=1)))
 
 # Prediction: Get min distance neighbors
 values, indices = tf.nn.top_k(distance, k=tf.cast(K,tf.int32), sorted=False)
 nearest_neighbors = tf.gather(tf.cast(ytr,tf.int32), indices)
 
 # nearest_neighbors = []
 # for i in range(K):
 #     nearest_neighbors.append(tf.argmax(ytr[indices[i]], 0))
 
 #nearest_neighbors = tf.map_fn(lambda x: tf.cast(tf.argmax(ytr[x], 0),tf.int32), indices)
 
 #neighbors_tensor = tf.pack(nearest_neighbors)
 
 #get the most repeated neighbor
 y, idx, count = tf.unique_with_counts(nearest_neighbors)
 
 #pred = y[argmax(count,0)]
 pred = tf.slice(y, begin=[tf.cast(tf.argmax(count, 0),tf.int32)], size=tf.constant([1]), name='pred')
 
 chosen_values = tf.gather(values, tf.where(tf.equal(tf.cast(nearest_neighbors, tf.float32),tf.cast(pred, tf.float32))))
 min_dist = tf.neg(tf.reduce_max(chosen_values), name='dist')
 
 
 # Save Graph
 with tf.Session() as sess:
 tf.train.write_graph(sess.graph_def, '.','kNN.pb', False)
 print("Graph Saved.")
*/
