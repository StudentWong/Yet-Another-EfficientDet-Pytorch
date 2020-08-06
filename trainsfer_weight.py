from backbone import EfficientDetBackbone

if __name__ == "__main__":
    model = EfficientDetBackbone(compound_coef=0, num_classes=len(obj_list),
                                 ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()