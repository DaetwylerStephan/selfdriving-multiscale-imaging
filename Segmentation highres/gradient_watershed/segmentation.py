# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 21:37:54 2023

@author: fyz11

helper functions for applying several standard DL segmentation models. 
"""

# import cupy 
import numpy as np 


def _ma_average(sig, winsize=3, mode='reflect', avg_func=np.nanmean):
    
    sig_ = np.pad(sig, [winsize//2, winsize//2], mode=mode)
    sig_out = []
    for ii in np.arange(len(sig)):
        data = sig_[ii:ii+winsize]
        sig_out.append(avg_func(data))
    return np.hstack(sig_out)
    
def _histeq_low_contrast(im, kernel_size=(256,256), clip_limit=0.01, fraction_threshold=0.1):
    
    import skimage.exposure as skexposure
    
    if skexposure.is_low_contrast(im, fraction_threshold):
        return im
    else:
        try:
            from .gpu import cuda_equalize_adapthist
            return cuda_equalize_adapthist(im, kernel_size=kernel_size, clip_limit=clip_limit)
        except:
            return skexposure.equalize_adapthist(im, kernel_size=kernel_size, clip_limit=clip_limit)

# def apply_cellpose_model_2D(im_stack, model, 
#                             model_channels, 
#                             best_diam=None, 
#                             model_invert=False, 
#                             test_slice=None, 
#                             diam_range=np.arange(15,51,5), 
#                             ksize=25,
#                             smoothwinsize=5,
#                             hist_norm=True, 
#                             kernel_size=(256,256),
#                             clip_limit=0.01,
#                             fraction_threshold=0.1,
#                             n_proc=48):
    
#     from .filters import var_filter
#     from tqdm import tqdm 
#     import pylab as plt 
#     from skimage.filters.rank import entropy
#     import skimage.morphology as skmorph
#     import skimage.exposure as skexposure
    
#     if best_diam is None:
#         diam_range = np.hstack(diam_range)
        
#         if test_slice is None:
#             # don't auto into the mid slice ---> instead pick the slice with most signal. 
#             if len(im_stack.shape)>3:
#                 signal_stack = [np.nanmedian(np.max(im_slice,axis=-1)) for im_slice in im_stack]
#             else:
#                 print('single channel')
#                 signal_stack = [np.nanmedian(im_slice) for im_slice in im_stack]
#             # test_slice = len(im_stack) // 2
#             test_slice = np.argmax(signal_stack)
#             print(test_slice, len(im_stack))
        
#         diam_score = []
        
#         for diam in diam_range[:]: 
            
#             img = im_stack[test_slice].copy()
#             if hist_norm:
#                 # this messes up ? 
#                 img = _histeq_low_contrast(img, kernel_size=kernel_size, clip_limit=clip_limit, fraction_threshold=fraction_threshold)
#                 # img = normalize(img, pmin=2, pmax=99.8, clip=True)
#                 img = skexposure.rescale_intensity(img)
#             mask, flow, style, diam = model.eval([img], 
#                                                 channels=model_channels, 
#                                                 batch_size=32,
#                                                 do_3D=False, 
#                                                 flow_threshold=0.6,
#                                                 diameter=diam, # this is ok 
#                                                 invert=model_invert) # try inverting?  
        
#             # score the content! e.g. sobel, var, 
#             # prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)+var_filter(flow[0][1][1], ksize=ksize))
#             prob = flow[0][2]
#             prob = 1./(1+np.exp(-prob))
#             # prob_score = np.nanmean(1./(var_filter(prob, ksize=ksize) + 0.1)*prob)
#             # prob_score = np.nanmean(var_filter(prob, ksize=ksize))
#             # prob_score = np.median(prob) / np.std(prob)
#             # prob_score = np.nanmean(entropy(prob, skmorph.disk(ksize)))
#             # prob_score = np.mean(prob) / (np.std(prob)) # signal to noise ratio. 
#             # prob_score = np.nanmean(entropy(flow[0][1][0]/5., skmorph.disk(ksize))) + np.nanmean(entropy(flow[0][1][1]/5., skmorph.disk(ksize)))
#             prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)+var_filter(flow[0][1][1], ksize=ksize)) # doesn't work weel 
#             # prob_score = np.nanmedian(var_filter(flow[0][1][0], ksize=ksize))+np.nanmedian(var_filter(flow[0][1][1], ksize=ksize))
            
#             # prob_score = mask[0].max()
#             # prob_score = np.nanmean(flow[0][2])
#             diam_score.append(prob_score)
            
#             plt.figure(figsize=(5,5))
#             plt.subplot(311)
#             plt.title(str(diam)+ ' '+str(prob_score))
#             plt.imshow(img[:1024,:1024])
#             plt.subplot(312)
#             plt.title(np.mean(flow[0][2]))
#             plt.imshow(flow[0][2][:1024,:1024])
#             plt.subplot(313)
#             plt.imshow(var_filter(flow[0][2], ksize=ksize)[:1024,:1024])
#             plt.show()
            
#         diam_score = np.hstack(diam_score)
        
#         # smooth this. 
#         diam_score = _ma_average(diam_score, winsize=smoothwinsize)
            
#         # =============================================================================
#         #     Compute the best. diameter for this view. 
#         # =============================================================================
#         best_diam = diam_range[np.argmax(diam_score)]
        
#         plt.figure()
#         plt.plot(diam_range, diam_score, 'o-')
#         plt.show()
        
#     print('auto determine cell diameter: ', best_diam)

#     import dask
#     from dask.distributed import Client
    
#     # # n_proc = 48
#     # client = Client(n_workers=n_proc)
#     client = Client()
#     print(client.cluster)
    
#     def run_cp_model(im_, model, channels, batch_size, do_3D, flow_threshold, diameter, invert):
        
#         mask, flow, style, diam = model.eval([im_], 
#                                             channels=model_channels, 
#                                             batch_size=batch_size,
#                                             do_3D=do_3D, 
#                                             flow_threshold=flow_threshold,
#                                             diameter=diameter, # this is ok 
#                                             invert=invert) # try inverting? 
        
#         return np.concatenate([mask[0][None,...], 
#                                flow[0][1], 
#                                flow[0][2][None,...]], axis=0).astype(np.float32)
    
#     # lazy_model = dask.delayed(model.eval)
#     lazy_model = dask.delayed(run_cp_model)
    
    
#     client.scatter(im_stack)
#     future = client.map(lazy_model, im_stack, 
#                                 model=model,
#                                 channels=model_channels, 
#                                 batch_size=32,
#                                 do_3D=False, 
#                                 flow_threshold=0.6,
#                                 diameter=best_diam, # this is ok 
#                                 invert=model_invert)
#     return client.gather(future)
    
#     # # with Client() as client:
#     # # all_masks = []
#     # # all_probs = []
#     # # all_flows = []
#     # # all_styles = []
#     # # all_diams = []
#     # all_results = []
#     # # can we use dask for this? and do cpu .
#     # # for zz in tqdm(np.arange(len(im_stack))):
#     # for zz in np.arange(len(im_stack)):
        
#     #     # mask, flow, style, diam = model.eval([im_stack[zz]], 
#     #     #                                         channels=model_channels, 
#     #     #                                         batch_size=32,
#     #     #                                         do_3D=False, 
#     #     #                                         flow_threshold=0.6,
#     #     #                                         diameter=best_diam, # this is ok 
#     #     #                                         invert=model_invert) # try inverting?  
#     #     # res = lazy_model.eval([im_stack[zz]], 
#     #     #                         channels=model_channels, 
#     #     #                         batch_size=32,
#     #     #                         do_3D=False, 
#     #     #                         flow_threshold=0.6,
#     #     #                         diameter=best_diam, # this is ok 
#     #     #                         invert=model_invert) # try inverting?  
        
#     #     res = lazy_model.eval(im_stack[zz].astype(np.float32), 
#     #                             channels=model_channels, 
#     #                             batch_size=32,
#     #                             do_3D=False, 
#     #                             flow_threshold=0.6,
#     #                             diameter=best_diam, # this is ok 
#     #                             invert=model_invert) # try inverting?  
        
#     #     # all_masks.append(mask[0])
#     #     # all_probs.append(flow[0][2])
#     #     # all_flows.append(flow[0][1])
#     #     # all_styles.append(style[0])
#     #     # all_diams.append(diam)
#     #     all_results.append(res)
        
#     # # all_masks = np.array(all_masks, dtype=np.uint16)
#     # # all_probs = np.array(all_probs, dtype=np.float32)
#     # # all_flows = np.array(all_flows, dtype=np.float32)
#     # # all_styles = np.array(all_styles, dtype=np.float32)
#     # # all_diams = np.array(all_diams, dtype=np.float32)
#     # all_results = dask.compute(all_results)
    
#     print('done')
#     print(len(all_results))
#     #     # all_masks, all_probs, all_flows, all_styles, all_diams = dask.compute(all_masks, all_probs, all_flows, all_styles, all_diams)
#     #     # results = dask.compute(*results)
#     #     # print(len(all_masks))
#     #     # client.close()
#     #     all_masks = np.array(all_masks, dtype=np.uint16)
#     #     all_probs = np.array(all_probs, dtype=np.float32)
#     #     all_flows = np.array(all_flows, dtype=np.float32)
#     #     all_styles = np.array(all_styles, dtype=np.float32)
#     #     all_diams = np.array(all_diams, dtype=np.float32)
    
#     # # client.close()
#     return all_results
#     # return (diam_range, diam_score, best_diam), (all_masks, all_probs, all_flows, all_styles, all_diams)

def apply_cellpose_model_2D_prob(im_stack, model, 
                            model_channels, 
                            best_diam=None, 
                            model_invert=False, 
                            test_slice=None, 
                            diam_range=np.arange(15,51,5), 
                            ksize=25,
                            smoothwinsize=5,
                            hist_norm=True, 
                            kernel_size=(256,256),
                            clip_limit=0.01,
                            fraction_threshold=0.1,
                            n_proc=48):
    
    from .filters import var_filter
    from .gpu import cuda_equalize_adapthist
    from tqdm import tqdm 
    import pylab as plt 
    from skimage.filters.rank import entropy
    import skimage.morphology as skmorph
    import skimage.exposure as skexposure
    
    if best_diam is None:
        diam_range = np.hstack(diam_range)
        
        if test_slice is None:
            # don't auto into the mid slice ---> instead pick the slice with most signal. 
            if len(im_stack.shape)>3:
                signal_stack = [np.nanmedian(np.max(im_slice,axis=-1)) for im_slice in im_stack]
            else:
                print('single channel')
                signal_stack = [np.nanmedian(im_slice) for im_slice in im_stack]
            # test_slice = len(im_stack) // 2
            test_slice = np.argmax(signal_stack)
            print(test_slice, len(im_stack))
        
        diam_score = []
        
        for diam in diam_range[:]: 
            
            img = im_stack[test_slice].copy()
            if hist_norm:
                # this messes up ? 
                img = _histeq_low_contrast(img, kernel_size=kernel_size, clip_limit=clip_limit, fraction_threshold=fraction_threshold)
                # img = normalize(img, pmin=2, pmax=99.8, clip=True)
                img = skexposure.rescale_intensity(img)
            _, flow, style = model.cp.eval([img], 
                                        channels=model_channels, 
                                        batch_size=32,
                                        do_3D=False, 
                                        flow_threshold=0.6,
                                        diameter=diam, # this is ok 
                                        invert=model_invert) # try inverting?  
        
            # score the content! e.g. sobel, var, 
            # prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)+var_filter(flow[0][1][1], ksize=ksize))
            prob = flow[0][2]
            prob = 1./(1+np.exp(-prob))
            # prob_score = np.nanmean(1./(var_filter(prob, ksize=ksize) + 0.1)*prob)
            # prob_score = np.nanmean(var_filter(prob, ksize=ksize))
            # prob_score = np.median(prob) / np.std(prob)
            # prob_score = np.nanmean(entropy(prob, skmorph.disk(ksize)))
            # prob_score = np.mean(prob) / (np.std(prob)) # signal to noise ratio. 
            # prob_score = np.nanmean(entropy(flow[0][1][0]/5., skmorph.disk(ksize))) + np.nanmean(entropy(flow[0][1][1]/5., skmorph.disk(ksize)))
            prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)+var_filter(flow[0][1][1], ksize=ksize)) # doesn't work weel 
            # prob_score = np.nanmedian(var_filter(flow[0][1][0], ksize=ksize))+np.nanmedian(var_filter(flow[0][1][1], ksize=ksize))
            
            # prob_score = mask[0].max()
            # prob_score = np.nanmean(flow[0][2])
            diam_score.append(prob_score)
            
            plt.figure(figsize=(5,5))
            plt.subplot(311)
            plt.title(str(diam)+ ' '+str(prob_score))
            plt.imshow(img[:1024,:1024])
            plt.subplot(312)
            plt.title(np.mean(flow[0][2]))
            plt.imshow(flow[0][2][:1024,:1024])
            plt.subplot(313)
            plt.imshow(var_filter(flow[0][2], ksize=ksize)[:1024,:1024])
            plt.show()
            
        diam_score = np.hstack(diam_score)
        
        # smooth this. 
        diam_score = _ma_average(diam_score, winsize=smoothwinsize)
            
        # =============================================================================
        #     Compute the best. diameter for this view. 
        # =============================================================================
        best_diam = diam_range[np.argmax(diam_score)]
        
        plt.figure()
        plt.plot(diam_range, diam_score, 'o-')
        plt.show()
        
    print('auto determine cell diameter: ', best_diam)

    all_probs = []
    all_flows = []
    all_styles = []
    
    
    for zz in tqdm(np.arange(len(im_stack))):
        img = im_stack[zz].copy()
        
        if hist_norm:
            # this messes up ? 
            img = _histeq_low_contrast(img, kernel_size=kernel_size, clip_limit=clip_limit, fraction_threshold=fraction_threshold)
            # img = normalize(img, pmin=2, pmax=99.8, clip=True)
            img = skexposure.rescale_intensity(img)
            
        _, flow, style = model.cp.eval([img],
                                       batch_size=32,
                                             channels=model_channels, 
                                             diameter=best_diam,
                                             model_loaded=True,
                                             invert=model_invert,
                                             compute_masks=False)
        all_probs.append(flow[0][2])
        all_flows.append(flow[0][1])
        all_styles.append(style[0])
        
    # _, all_flows, all_styles = model.cp.eval(im_stack, 
    #                                         batch_size=32,
    #                                               channels=model_channels, 
    #                                               diameter=best_diam,
    #                                               model_loaded=True,
    #                                               invert=model_invert,
    #                                               compute_masks=False)

    # all_probs = all_flows[2].copy()
    # all_flows = all_flows[1].copy()
    
    all_probs = np.array(all_probs, dtype=np.float32)
    all_flows = np.array(all_flows, dtype=np.float32)
    all_styles = np.array(all_styles, dtype=np.float32)

    return (diam_range, diam_score, best_diam), (all_probs, all_flows, all_styles)