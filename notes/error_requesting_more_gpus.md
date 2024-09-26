
sq
Showing results for user adamlesnikowski
Currently 2 running jobs and 1 pending job (most recent job first):
+----------+------+-------------+---------------+-----------------+---------+---------+------------+
|  Job ID  | Name |   Account   |     Nodes     |       QOS       |  Time   |  State  |   Reason   |
+----------+------+-------------+---------------+-----------------+---------+---------+------------+
| 21189509 | zsh  | ac_holliday | 1x savio3_gpu | a40_gpu3_normal |  0:00   | PENDING | QOSGrpGRES |
| 21188237 | zsh  | ac_holliday | 1x savio3_gpu | a40_gpu3_normal | 3:24:11 | RUNNING |            |
| 21188236 | zsh  | ac_holliday | 1x savio3_gpu | a40_gpu3_normal | 3:25:41 | RUNNING |            |
+----------+------+-------------+---------------+-----------------+---------+---------+------------+

21189509:
 - This job is waiting until the QOS a40_gpu3_normal has available resources.
   QOS resource limit: 128 cpus, 16 gres/gpus
   QOS currently using: 112 cpus: 21189510 (16 cpus), 21189507 (16 cpus), 21189505 (16 cpus), 21188490 (16 cpus), 21168663 (16 cpus), 21133101 (16 cpus), 21188237 (8 cpus), 21188236 (8 cpus)
   This job is requesting: 8 cpus
   You may consider submitting with these other QOS: gtx2080_gpu3_normal, savio_lowprio, v100_gpu3_normal
