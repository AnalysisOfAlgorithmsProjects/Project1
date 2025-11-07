📘 Analysis of Algorithms — Project 1

Author: Saranya Yadlapalli, Dinesh Kollipakla
Course: Analysis of Algorithms (AOA) COT 5405 
Institution: University of Florida

🧠 Project Overview

This project demonstrates two fundamental algorithmic paradigms — Greedy and Divide and Conquer — applied to real-world computer science problems.
The goal is to analyze how these paradigms differ in structure, performance, and problem-solving strategies through theoretical and experimental evaluation.

🔹 Part I — Greedy Algorithm (Huffman Coding)

Objective:
Implement Huffman Coding to compress PDF/PostScript content streams using a greedy selection strategy that minimizes average code length.

Key Features:

Builds optimal prefix-free Huffman codes based on symbol frequency.

Achieves efficient compression with canonical code representation.

Compression ratio improvement: ≈ 39–44 %.

Theoretical complexity: O(n log n).

🔹 Part II — Divide and Conquer Algorithm (Load Balancing)

Objective:
Design a recursive Divide and Conquer algorithm to distribute tasks evenly across multiple servers in a distributed system.

Key Features:

Recursively partitions task and server sets.

Balances workloads locally and rebalances only when beneficial.

Ensures global balance within 10 % imbalance threshold.

Theoretical complexity: O(n log n); confirmed experimentally.


📊 Experimental Results
Algorithm	Metric	Observed Trend
Huffman Coding	Compression Ratio	39–44 % improvement
Load Balancer	Runtime	Near-linear (O(n log n))
Load Balancer	Imbalance	≤ 10 % threshold

All results validated the analytical complexity and algorithmic design principles.

🧮 Tools and Libraries

Language: Python 3.12


⚙️ How to Run
🟩 1. Run Greedy Huffman Compression
cd greedy
python3 huffman_compression.py

🟦 2. Run Divide and Conquer Load Balancer
cd divide_and_conquer
python3 load_balancer.py


This will generate runtime graphs automatically and display them using Matplotlib.

Libraries: matplotlib, random, time, csv

Documentation: LaTeX (IEEE/ACM format)
