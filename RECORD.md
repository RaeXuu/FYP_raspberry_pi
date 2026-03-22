 main_pi.py：先切（环形缓冲区每次取最新 2s window），再对每个 window 单独归一化                                                                                              
                                                                                                                                                                              
  main_pi_flow.py：去掉整块归一化。在滑窗循环内、mel 计算前，对每个 window 单独做峰值归一化   
  原始int16                                                                                                                                                                                  
                  
  main_pi_debug.py via preprocess_wav_for_pi：load_wav 先对整个文件归一化，再切片      

  record_debug.py：先按 2s 无 overlap 切片，再对每个 segment 单独归一化（传入 preprocess_array_for_pi）                                                                                           
                  
  ---                                                                                                                                                                         
  训练时走的是 load_wav → segment_audio，所以训练的做法是先对整段录音归一化，再切片。