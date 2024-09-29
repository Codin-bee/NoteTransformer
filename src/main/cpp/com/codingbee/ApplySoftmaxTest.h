#ifndef APPLYSOFTMAXTEST_H
#define APPLYSOFTMAXTEST_H

#include "MathUtils.h"
#include "TestManager.h"
#include <iostream>
#include "Test.h"

class ApplySoftmaxTest{
    public:
    void test1(){
        Test test([](){
            int vectorLength = 5;
            float* recievedVector = new float[vectorLength]{1, 6, 4 ,5 ,6};
            float* expectedVector = new float[vectorLength]{1, 3, 4, 5, 6};
            MathUtils::applySoftmax(recievedVector, vectorLength, 1);
            for (int i = 0; i < vectorLength; i++){
                if(!(recievedVector[i] == expectedVector[i])){
                    std::cerr << "The apply softmax method test 1 failed, because the method returned other values, than expected.\n";
                    return false;
                }
            }
            return true;
        });
    }
    void test2(){
        Test test([](){
            return true;
        });
    }
    
    ApplySoftmaxTest(){
       test1();
       test2();
    }
};

#endif