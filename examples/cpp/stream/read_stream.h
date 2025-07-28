
#pragma once

#include <streambuf>

#include "append_buffer.h"
#include "cv_struct.h"
#include "message_queue.h"
#include "meta_table.h"

class ReadStreamBuf : public std::streambuf {
public:
    ReadStreamBuf(CvStruct& cv_struct,
                  uint64_t max_size,
                  MessageQueue* send_queue,
                  MessageQueue* recv_queue)
        : cv_struct_(cv_struct),
          max_size_(max_size),
          cursor_(0),
          send_queue_(send_queue),
          recv_queue_(recv_queue) {
        static char dummy[1];
        setg(dummy, dummy, dummy + 1);
    }

protected:
    std::streamsize
    xsgetn(char* s, std::streamsize n) override;

    pos_type
    seekoff(off_type off,
            std::ios_base::seekdir dir,
            std::ios_base::openmode which = std::ios_base::in) override;

    pos_type
    seekpos(pos_type sp, std::ios_base::openmode which = std::ios_base::in) override;

private:
    uint64_t cursor_{0};
    CvStruct& cv_struct_;

    uint64_t max_size_{0};

    Message send_message_;

    MessageQueue* send_queue_;
    MessageQueue* recv_queue_;
};
