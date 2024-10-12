#ifndef TEST_H
#define TEST_H

#include "TestManager.h"
#include "ApplySoftmaxTest.h"

#include <functional>

class Test{
    public:
    Test(std::function<bool()> test){
        TestManager::registerTest(test);
    }
};

#endif