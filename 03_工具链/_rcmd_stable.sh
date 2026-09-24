set -u
F=$(ls -1d /root/ops/fp51full_* | tail -1)/train.log
N=$(ls -1d /root/ops/fp51neg_* | tail -1)/train.log
echo "full: $(ls -l $F | awk '{print $5}') B  sha=$(sha256sum $F | cut -c1-12)"
echo "neg : $(ls -l $N | awk '{print $5}') B  sha=$(sha256sum $N | cut -c1-12)"
sleep 4
echo "4 秒后："
echo "full: $(ls -l $F | awk '{print $5}') B  sha=$(sha256sum $F | cut -c1-12)"
echo "neg : $(ls -l $N | awk '{print $5}') B  sha=$(sha256sum $N | cut -c1-12)"
echo "还有没有相关进程：$(pgrep -c -f 'trai''ner.py' || true) $(pgrep -c -f 'torch''run' || true)"
echo "DONE_STABLE"
