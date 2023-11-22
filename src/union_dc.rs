/*
 * File: /union_dc.rs
 * Created Date: Wednesday November 22nd 2023
 * Author: Zihan
 * -----
 * Last Modified: Thursday, 23rd November 2023 1:18:50 am
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 */

use std::collections::VecDeque;

pub struct UnionDC {}

trait UnionDCImpl<T> {
    fn union_dc(
        queue: &mut VecDeque<T>,
        func: impl Fn(T, T) -> T,
        end_condition: impl Fn(&VecDeque<T>) -> bool,
    );
}

impl UnionDC {
    pub fn union_dc<T, F>(
        queue: &mut VecDeque<T>,
        mut func: F,
        end_condition: impl Fn(&VecDeque<T>) -> bool,
    )
    where
        F: FnMut(T, T) -> T,
    {
        while !end_condition(queue) {
            let a = queue
                .pop_front()
                .expect("Queue should have enough elements");
            let b = queue
                .pop_front()
                .expect("Queue should have enough elements");
            let result = func(a, b);
            queue.push_back(result);
        }
        // queue
    }
}
