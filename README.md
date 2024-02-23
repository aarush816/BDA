# BDA

hadoop jar '/home/hdoop/hadoop-streaming-3.3.6.jar' -file ./mapwc.py -mapper ./mapwc.py -file ./redwc.py -reducer ./redwc.py -input /066_bda/input.txt -output /066_bda/oup3

hdfs dfs -cat /066_bda/oup3/part-00000

hdfs dfs -get /066_bda/oup3/part-00000 /home/hdoop

Cat part-00000
