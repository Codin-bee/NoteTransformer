#ifndef TESTINGMANAGER_H
#define TESTINGMANAGER_H

#include <vector>
#include <functional>

class TestingManager{
    private:
    std::vector<std::function<void()>> tests;
    public:
    void registerTest(std::function<void()> test){
        tests.push_back(test);
    }
    void runAllTests();
};
#endif