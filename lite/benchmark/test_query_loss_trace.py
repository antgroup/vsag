# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import unittest
from verify_query_loss_trace import classify


class LossTraceTests(unittest.TestCase):
    def row(self,visited,formal,rerank):
        return dict(visited=str(visited),formal=str(formal),visited_rerank=str(rerank))
    def test_exclusive_accounting(self):
        result=classify([self.row(1,1,1),self.row(0,0,0),self.row(1,0,1),self.row(1,0,0)])
        self.assertEqual(result,dict(hits=1,unvisited=1,shortlist_loss=1,visited_rank_loss=1))
    def test_formal_unvisited_rejected(self):
        with self.assertRaises(AssertionError): classify([self.row(0,1,0)])
    def test_rerank_unvisited_rejected(self):
        with self.assertRaises(AssertionError): classify([self.row(0,0,1)])
    def test_nonboolean_rejected(self):
        for values in [(2,0,0),(1,2,0),(1,0,2)]:
            with self.subTest(values=values),self.assertRaises(AssertionError): classify([self.row(*values)])
    def test_formal_hit_does_not_double_count(self):
        self.assertEqual(classify([self.row(1,1,0)])['hits'],1)
    def test_empty_accounting(self):
        self.assertEqual(sum(classify([]).values()),0)


if __name__=='__main__': unittest.main()
