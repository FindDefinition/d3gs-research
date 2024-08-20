import torch 

def main():
    start_ev = torch.mps.Event(enable_timing=True)
    end_ev = torch.mps.Event(enable_timing=True)
    start_ev.record()
    end_ev.record()
    # torch.mps.synchronize()
    # TODO sync event will hang
    # start_ev.synchronize()
    # end_ev.synchronize()
    duration = start_ev.elapsed_time(end_ev)


if __name__ == "__main__":
    main()
