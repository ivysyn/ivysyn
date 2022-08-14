#!/bin/bash
sudo mkdir /mnt/tensorflow-ivysyn
sudo mkdir /mnt/pytorch-ivysyn
sudo mount -t tmpfs tmpfs /mnt/tensorflow-ivysyn
sudo mount -t tmpfs tmpfs /mnt/pytorch-ivysyn
sudo chown -R ivyusr:ivyusr /mnt/tensorflow-ivysyn
sudo chown -R ivyusr:ivyusr /mnt/pytorch-ivysyn
