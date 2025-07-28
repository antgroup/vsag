
#include "record_write_stream.h"

std::streamsize
RecordWriteStreamBuf::xsputn(const char* s, std::streamsize n) {
    if (n == 0) {
        return n;
    }
    if (cv_struct_.pause_signal_.load()) {
        std::unique_lock<std::mutex> lk(cv_struct_.mu_);
        cv_struct_.cv_.wait(lk, [this] { return cv_struct_.pause_signal_.load() == false; });
    }
    MetaItem meta_item;
    meta_item.offset = cursor_;
    meta_item.size = static_cast<uint64_t>(n);
    if (n > 2048) {
        meta_item.addr = const_cast<char*>(s);
    } else {
        auto* new_addr = append_buffer_->Append(s, n);
        meta_item.addr = const_cast<char*>(new_addr);
    }
    meta_table_->InsertMetaItem(meta_item);
    cursor_ += static_cast<uint64_t>(n);
    total_size_ += static_cast<uint64_t>(n);
    return n;
}

RecordWriteStreamBuf::pos_type
RecordWriteStreamBuf::seekoff(off_type off,
                              std::ios_base::seekdir dir,
                              std::ios_base::openmode which) {
    if (which != std::ios_base::out) {
        return pos_type(off_type(-1));
    }

    switch (dir) {
        case std::ios_base::beg:
            cursor_ = static_cast<uint64_t>(off);
            break;
        case std::ios_base::cur:
            if (off >= 0) {
                cursor_ += static_cast<uint64_t>(off);
            } else {
                if (static_cast<uint64_t>(-off) > cursor_) {
                    return pos_type(off_type(-1));
                }
                cursor_ -= static_cast<uint64_t>(-off);
            }
            break;
        case std::ios_base::end:
            return pos_type(off_type(-1));
        default:
            return pos_type(off_type(-1));
    }

    return pos_type(static_cast<off_type>(cursor_));
}

RecordWriteStreamBuf::pos_type
RecordWriteStreamBuf::seekpos(pos_type sp, std::ios_base::openmode which) {
    if (which != std::ios_base::out) {
        return pos_type(off_type(-1));
    }

    if (sp < 0) {
        return pos_type(off_type(-1));
    }

    cursor_ = static_cast<uint64_t>(sp);
    return pos_type(static_cast<off_type>(cursor_));
}
