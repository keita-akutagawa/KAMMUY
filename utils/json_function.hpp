#ifndef JSON_FUNCTION_HPP 
#define JSON_FUNCTION_HPP 

#include <nlohmann/json.hpp>
#include <string>
#include <stdexcept>
#include <iostream>
#include <initializer_list>


template <typename T>
T getRequiredForJSON(
    const nlohmann::json& j, 
    std::initializer_list<const char*> path
)
{
    const nlohmann::json* cur = &j;
    std::string fullpath;

    for (const char* key : path) {
        if (!cur->is_object() || !cur->contains(key)) {
            if (!fullpath.empty()) fullpath += ".";
            fullpath += key;
            throw std::runtime_error(
                "Missing required JSON key: " + fullpath
            );
        }

        if (!fullpath.empty()) fullpath += ".";
        fullpath += key;
        cur = &cur->at(key);
    }

    try {
        return cur->get<T>();
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(
            "Type error at JSON key: " + fullpath + 
            " (" + e.what() + ")"
        );
    }
}


template <typename T>
T getRequiredForJSON(
    const nlohmann::json& j,
    const char* key
)
{
    std::initializer_list<const char*> path{key};
    return getRequiredForJSON<T>(j, path);
}

#endif
