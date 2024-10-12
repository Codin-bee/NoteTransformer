#include "TestManager.h"
#include <iostream>
std::vector<std::function<bool()>> TestManager::tests;
void TestManager::runAllTests(){
    int testsWorking = 0;
    int totalTests = 0;
    for(const auto& test : tests){
        totalTests++;
        if (test()){
            testsWorking++;
        }
    }
    std::cout << "Tests in total: " << totalTests << "\n"
    << "Working tests: " << testsWorking << "\n"
    << "Tests not working: " << totalTests - testsWorking << "\n";
}
void TestManager::registerTest(std::function<bool()> test){
        tests.push_back(test);
    }