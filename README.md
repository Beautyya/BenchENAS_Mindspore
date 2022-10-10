- The codes have been tested on Python 3.7 + mindspore 1.7.1 + mindvision 0.1.0 (mindspore >= 1.5.3 seems also ok, but not test thoroughly)



- #Requirements:

 - Center Computer:
   - redis (ubuntu software, start using the command redis-server --protected-mode on)
   - sshpass (python lib)

 - Conter Computer & workers:
   - multiprocess (python lib)
   - redis (python lib)
 
 
- # How to use
- Start the redis-server on the center computer (redis-server --protected-mode no)
- Start the init_compute.py script to start the compute platform
- Start the algorithm you would like to perform
