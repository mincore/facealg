/* ===================================================
 * Copyright (C) 2017 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: nypack.h
 *     Created: 2017-07-24 18:48
 * Description:
 * ===================================================
 */
#ifndef _NYPACK_H
#define _NYPACK_H

#include <string>
#include <vector>

bool NYGetKey(std::string &key);

class NYUnpacker {
public:
    NYUnpacker();
    ~NYUnpacker();
    bool load(const char *passphrase[3], const char *file);
    void list(std::vector<std::string> &names);
    bool read(const char *name, std::vector<char> &data);
private:
    void *impl_;
};

#endif
