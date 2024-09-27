#include "TestingManager.h"

void TestingManager::runAllTests(){
    for(const auto& test : tests){
        test();
    }
}