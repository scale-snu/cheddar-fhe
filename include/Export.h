#pragma once

#ifdef _WIN32
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT __attribute__((visibility("default")))
#endif
