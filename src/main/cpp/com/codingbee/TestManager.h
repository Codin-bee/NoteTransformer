#ifndef TESTMANAGER_H
#define TESTMANAGER_H

#include <vector>
#include <functional>

class TestManager{
    public:
    static std::vector<std::function<bool()>> tests;
    static void registerTest(std::function<bool()> test);
    static void runAllTests();
};
#endif