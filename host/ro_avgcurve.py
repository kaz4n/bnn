#!/usr/bin/env python3
"""Averaging-limit test: capture MANY single-shot sensor reads of one image, then plot
corr(cumulative-averaged sensor, window magnitude) vs number-of-averages N.
  corr still rising at large N  -> averaging-limited: a high-avg capture will help (cheap).
  corr plateaus early           -> dilution-limited: needs RTL floorplan/clock-gate or EM.
Run in cwenv. ~3-5 min."""
import argparse, time
import numpy as np
LINE=28; IMG=LINE*LINE
REG_IMAGE,REG_OUTPUT,REG_KERNEL,REG_GO,REG_STATUS,REG_SENSOR_CTRL=0,16,32,33,34,35

def pack_kernel(k):
    bits=(k.flatten()>0).astype(np.uint8); out=bytearray(); byte=nb=0
    for b in bits:
        byte=(byte<<1)|int(b); nb+=1
        if nb==8: out.append(byte); byte=nb=0
    if nb: out.append(byte<<(8-nb))
    return list(out)

def wait_idle(t,to=1.0):
    dl=time.time()+to
    while time.time()<dl:
        if (t.fpga_read(REG_STATUS,1)[0]&1)==0: return
        time.sleep(0.0005)

def winmag(img,K=3):
    p=K//2; m=np.zeros(IMG)
    for y in range(LINE):
        for x in range(LINE):
            s=0
            for i in range(K):
                for j in range(K):
                    yy,xx=y+i-p,x+j-p
                    if 0<=yy<LINE and 0<=xx<LINE: s+=abs(int(img[yy,xx]))
            m[y*LINE+x]=s
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bitstream",default="../build/cw305_bnn_ro3.bit")
    ap.add_argument("--kernels",default="../training/artifacts/model_3x3/layer1_kernels.npy")
    ap.add_argument("--images",default="mnist_test.npz")
    ap.add_argument("--img",type=int,default=0)
    ap.add_argument("--nreads",type=int,default=800)
    ap.add_argument("--freq",type=float,default=10e6)
    ap.add_argument("--sensor-reg",type=int,default=16)
    args=ap.parse_args()
    import chipwhisperer as cw
    xte=np.load(args.images)["images"]; img=xte[args.img].astype(np.uint8)
    kern=np.load(args.kernels)[0].astype(np.int8); packed=pack_kernel(kern)
    wm=winmag(img)
    target=cw.target(None,cw.targets.CW305,bsfile=args.bitstream,fpga_id="100t",force=True)
    try:
        target.pll.pll_enable_set(True); target.pll.pll_outenable_set(True,0); target.pll.pll_outenable_set(True,1)
        target.pll.pll_outfreq_set(args.freq,1)
        try: target.fpga_write(REG_SENSOR_CTRL,[1]); time.sleep(0.1)
        except Exception: pass
        target.fpga_write(REG_IMAGE,img.flatten().tolist())
        target.fpga_write(REG_KERNEL,packed)
        reads=np.zeros((args.nreads,IMG))
        for r in range(args.nreads):
            target.fpga_write(REG_GO,[1]); time.sleep(0.02); wait_idle(target)
            raw=bytes(target.fpga_read(args.sensor_reg,2*IMG))
            reads[r]=np.frombuffer(raw,dtype="<u2").astype(np.float64)
        print(f"img {args.img}: {args.nreads} reads captured")
        print(f"{'N':>5} {'corr':>8}")
        for N in [1,5,10,25,50,100,200,400,args.nreads]:
            if N>args.nreads: continue
            avg=reads[:N].mean(0)
            c=np.corrcoef(avg,wm)[0,1]
            print(f"{N:>5} {c:>8.3f}", flush=True)
    finally:
        target.dis()

if __name__=="__main__":
    main()
